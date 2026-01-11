# src/cavl_doc/trainers/curriculum_trainer.py

import os
import csv
import logging
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy

from cavl_doc.modules.losses import build_loss
from cavl_doc.trainers.rl_trainer import validate_siam_on_loader, AverageMeter

logger = logging.getLogger(__name__)

from torch.distributions import Categorical

class CurriculumTrainer:
    """
    Trainer that implements a Multi-Phase Curriculum Strategy combined with RL-based Data Selection.
    
    Curriculum (Macro-Strategy): Manages the transition of Loss Functions (e.g., Contrastive -> ExpFace -> Elastic).
    Professor (Micro-Strategy): Manages the selection of data samples (Hard Mining) within each phase using RL.
    """
    def __init__(
        self, 
        model,
        professor_model, # Added Professor
        tokenizer,
        train_loader, 
        val_loader, 
        device, 
        config,
        wandb_run=None
    ):
        self.model = model
        self.professor_model = professor_model # Store Professor
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.wandb = wandb_run
        
        self.output_dir = config.get('output_dir', 'checkpoints')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # State
        self.phase = 1
        self.current_epoch = 0
        self.best_val_eer = 1.0
        self.patience_counter = 0
        self.patience_limit = config.get('patience', 3)
        self.global_step = 0
        
        # RL State
        self.baseline = 0.0
        self.baseline_alpha = config.get('baseline_alpha', 0.01)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        
        # Configuration for Phases
        self.phases = []
        if config.get('phase1_loss'): self.phases.append(config['phase1_loss'])
        if config.get('phase2_loss'): self.phases.append(config['phase2_loss'])
        if config.get('phase3_loss'): self.phases.append(config['phase3_loss'])
        
        self.phase_idx = 0
        self.current_loss_name = self.phases[0]
        
        # Initial Setup
        self.criterion = self._build_criterion(self.current_loss_name)
        self.optimizer = self._build_optimizer()
        self.professor_optimizer = optim.Adam(
            self.professor_model.parameters(), 
            lr=config.get('professor_lr', 1e-4)
        )
        
        print(f"🎓 CurriculumTrainer Initialized with {len(self.phases)} phases.")
        print(f"   Phase 1/{len(self.phases)}: {self.current_loss_name} | Mining: RL-Professor")

    def _build_criterion(self, loss_type):
        print(f"   Building Loss: {loss_type}")
        return build_loss(
            loss_type,
            margin=self.config.get('margin', 0.5),
            s=self.config.get('scale', 64.0),
            num_classes=self.config.get('num_classes'),
            std=self.config.get('std', 0.05),
            in_features=self.config.get('projection_output_dim', 512)
        ).to(self.device)

    def _build_optimizer(self):
        # Parameters: Model + Loss Centers
        params = list(self.model.parameters()) + list(self.criterion.parameters())
        return optim.AdamW(
            params, 
            lr=self.config.get('lr', 1e-4), 
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

    def _transfer_loss_weights(self, old_criterion, new_criterion):
        """Transfers learned class centers from Phase 1 loss to Phase 2 loss."""
        if hasattr(old_criterion, 'weight') and hasattr(new_criterion, 'weight'):
            if old_criterion.weight.shape == new_criterion.weight.shape:
                print("   Transferring loss weights (class centers) to new objective...")
                new_criterion.weight.data = old_criterion.weight.data.clone()
            else:
                print("   Warning: Loss weight shapes mismatch. Cannot transfer.")

    def switch_to_next_phase(self):
        next_idx = self.phase_idx + 1
        if next_idx >= len(self.phases):
            return

        print("\n" + "="*40)
        print(f"🚀 SWITCHING TO PHASE {next_idx + 1}: {self.phases[next_idx]}")
        print("="*40 + "\n")
        
        self.phase_idx = next_idx
        self.current_loss_name = self.phases[next_idx]
        
        # 1. Build new Loss
        new_criterion = self._build_criterion(self.current_loss_name)
        
        # 2. Transfer weights if compatible
        prev_loss_name = self.phases[next_idx - 1]
        if prev_loss_name != 'contrastive':
            self._transfer_loss_weights(self.criterion, new_criterion)
        else:
            print("   Note: Previous loss was Contrastive (no centers). Initializing new centers.")
            
        self.criterion = new_criterion
        
        # 3. Rebuild Optimizer
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.5 
        print(f"   Reducing LR from {current_lr} to {new_lr}")
        
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=new_lr,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Reset patience
        self.patience_counter = 0 

    def rl_mining_step(self, img_a, img_b, labels, cls_a, cls_b):
        """
        Uses the Professor Network to select the best batch from the candidate pool.
        Returns: selected_indices, log_probs, entropy
        """
        batch_size = self.config.get('student_batch_size', 4)
        pool_size = len(img_a)
        
        if pool_size <= batch_size:
            return list(range(pool_size)), None, None

        # 1. Forward Pass (Student) to get State
        self.model.eval()
        self.professor_model.train()
        
        with torch.no_grad():
            def _fwd(img_list):
                embs = []
                for img in img_list:
                    img = img.unsqueeze(0).to(self.device)
                    embs.append(self.model(images=img))
                return torch.cat(embs, dim=0)
            
            ea = _fwd(img_a)
            eb = _fwd(img_b)
            
            # Calculate individual losses for State
            if self.current_loss_name in ['contrastive', 'angular']:
                 sl = self.criterion.forward_individual(ea, eb, labels.to(self.device))
            else:
                 la = self.criterion.forward_individual(ea, cls_a.to(self.device))
                 lb = self.criterion.forward_individual(eb, cls_b.to(self.device))
                 sl = (la + lb) / 2.0

            # Normalize State
            denom = (sl.max() - sl.min()).item() or 1.0
            sl_norm = (sl - sl.min()) / (denom + 1e-6)
            state = sl_norm.unsqueeze(-1) # [Pool, 1]

        # 2. Professor Prediction
        logits = self.professor_model(state).squeeze(-1)
        dist = Categorical(logits=logits)
        
        # 3. Sample Action
        idxs = dist.sample((batch_size,))
        log_probs = dist.log_prob(idxs)
        entropy = dist.entropy().mean()
        
        return idxs.tolist(), log_probs, entropy

    def train_epoch(self):
        self.model.train()
        avg_loss = AverageMeter()
        avg_rew = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Phase {self.phase_idx + 1} ({self.current_loss_name}) | Ep {self.current_epoch+1}", ncols=100)
        
        for batch_idx, (img_a, img_b, labels, cls_a, cls_b) in enumerate(pbar):
            # 1. RL Mining
            selected_idxs, log_probs, entropy = self.rl_mining_step(img_a, img_b, labels, cls_a, cls_b)
            
            # Filter batch
            sa = [img_a[i] for i in selected_idxs]
            sb = [img_b[i] for i in selected_idxs]
            s_cls_a = cls_a[selected_idxs].to(self.device)
            s_cls_b = cls_b[selected_idxs].to(self.device)
            s_labels = labels[selected_idxs].to(self.device)
            
            # 2. Training Step (Student)
            self.model.train()
            self.optimizer.zero_grad()
            
            def _fwd_train(img_list):
                embs = []
                for img in img_list:
                    img = img.unsqueeze(0).to(self.device)
                    embs.append(self.model(images=img))
                return torch.cat(embs, dim=0)

            sea = _fwd_train(sa)
            seb = _fwd_train(sb)
            
            # Student Loss
            if self.current_loss_name in ['contrastive', 'angular']:
                 loss = self.criterion(sea, seb, s_labels)
            else:
                 loss = self.criterion(sea, s_cls_a) + self.criterion(seb, s_cls_b)
            
            loss.backward()
            self.optimizer.step()
            
            # 3. Training Step (Professor)
            if log_probs is not None:
                with torch.no_grad():
                    # Calculate Reward (Loss on the selected batch)
                    if self.current_loss_name in ['contrastive', 'angular']:
                        s_ind = self.criterion.forward_individual(sea.detach(), seb.detach(), s_labels)
                    else:
                        l_a = self.criterion.forward_individual(sea.detach(), s_cls_a)
                        l_b = self.criterion.forward_individual(seb.detach(), s_cls_b)
                        s_ind = (l_a + l_b) / 2.0
                
                rew = s_ind.detach()
                avg_r = float(rew.mean().item())
                
                # Advantage
                adv = rew - self.baseline
                adv_std_val = adv.std(unbiased=False).clamp(min=1e-6)
                adv_norm = (adv - adv.mean()) / (adv_std_val + 1e-6)
                
                # Policy Gradient Update
                self.professor_optimizer.zero_grad()
                ploss = - (log_probs * adv_norm).mean() - self.entropy_coeff * entropy
                ploss.backward()
                self.professor_optimizer.step()
                
                # Update Baseline
                self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * avg_r
                avg_rew.update(avg_r)
            
            avg_loss.update(loss.item())
            self.global_step += 1
            
            pbar.set_postfix({'L': f"{avg_loss.avg:.4f}", 'R': f"{avg_rew.avg:.4f}"})
            
            if self.wandb:
                self.wandb.log({
                    "train/loss": loss.item(),
                    "train/reward": avg_rew.avg,
                    "train/phase": self.phase_idx + 1,
                    "step": self.global_step
                })
                
        return avg_loss.avg

    def run(self, epochs):
        print(f"🚀 Starting Curriculum Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_eer, val_thr, val_r1 = validate_siam_on_loader(
                self.model, self.val_loader, self.device, self.criterion
            )
            
            print(f"📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val EER:    {val_eer*100:.2f}%")
            print(f"   Val R@1:    {val_r1*100:.2f}%")
            
            if self.wandb:
                self.wandb.log({
                    "val/loss": val_loss,
                    "val/eer": val_eer,
                    "val/r1": val_r1,
                    "epoch": epoch + 1
                })
            
            # Check Phase Transition (Curriculum Logic)
            self._check_phase_transition(val_eer)

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}.pt")

    def _check_phase_transition(self, val_eer):
        """
        Decides when to switch to next phase.
        Logic: If validation EER plateaus (no improvement for 'patience' epochs), switch.
        """
        improvement = self.best_val_eer - val_eer
        
        if val_eer < self.best_val_eer:
            self.best_val_eer = val_eer
            self.patience_counter = 0
            self.save_checkpoint(f"best_phase{self.phase_idx + 1}.pt")
        else:
            self.patience_counter += 1
            print(f"   ⏳ Patience: {self.patience_counter}/{self.patience_limit}")
            
        # Trigger Switch
        if self.patience_counter >= self.patience_limit:
            if self.phase_idx < len(self.phases) - 1:
                print(f"   ⚠️ Performance plateaued in Phase {self.phase_idx + 1}.")
                self.switch_to_next_phase()
            else:
                print("   ⚠️ Plateau reached in final phase. Consider stopping.")

    def save_checkpoint(self, filename):
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.criterion.state_dict(),
            'phase': self.phase_idx + 1,
            'best_eer': self.best_val_eer
        }, path)
