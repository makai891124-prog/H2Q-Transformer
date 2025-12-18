import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
import numpy as np
import time
import os
import sys
import requests
import zipfile
import io

# ==========================================
# 0. H2Q-MicroStream: Unicode 通用编码版
# ==========================================
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    device_compute = torch.device("cuda")
    device_structure = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🌊 H2Q-MicroStream Online: {gpu_name}")
    print(f"   [Mode: Unicode Standard] [Batch: 24] [Rank: 8]")
else:
    device_compute = torch.device("cpu")
    device_structure = torch.device("cpu")

CONFIG = {
    'dim': 768,           
    'factor_size': 32,
    'fixed_rank': 8,       
    'depth': 12,           
    'seq_len': 128,        
    'batch_size': 24,      
    'grad_accum_steps': 1, 
    'lr': 6e-4,            
    'weight_decay': 0.02,
    'dropout_rate': 0.1,
    'eval_interval': 200,   
    'axiom_lambda': 0.1,
    'total_iters': 200000, 
    'checkpoint_path': 'h2q_unicode.pt',
    'best_model_path': 'h2q_unicode_best.pt',
    'data_dir': 'data_wikitext',
}

# ==========================================
# 1. 顺序流加载器 (Unicode 标准化)
# ==========================================
class SequentialLoader:
    def __init__(self, block_size, batch_size, data_dir, split='train'):
        self.block_size = block_size
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        self.train_path = os.path.join(data_dir, 'wiki.train.raw')
        self.val_path = os.path.join(data_dir, 'wiki.valid.raw')
        self._prepare_data()
        
        path = self.train_path if split == 'train' else self.val_path
        print(f"🌊 初始化 {split} 流式加载器 (Unicode Mode)...")
        with open(path, 'r', encoding='utf-8') as f: text = f.read()
        
        # 🔥 核心修改：不再动态生成字典，而是使用固定的 Unicode 映射
        # 词表大小固定为 256 (Byte-level 覆盖)
        self.vocab_size = 256
        
        # 数据转换：直接使用 ord(c) 获取 Unicode 编码
        # 如果超过 255 (非 Latin-1 字符)，映射为 0 (UNK)
        data_list = []
        for c in text:
            code = ord(c)
            if code < 256:
                data_list.append(code)
            else:
                data_list.append(0) # 未知字符/非ASCII字符
        
        data = torch.tensor(data_list, dtype=torch.long)
        
        # 重塑为 (Batch_Size, N) - 形成并行阅读流
        num_batches = len(data) // batch_size
        data = data[:num_batches * batch_size]
        self.data = data.view(batch_size, num_batches).contiguous().to(device_compute)
        
        self.num_batches = num_batches
        self.current_idx = 0
        
    def _prepare_data(self):
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        if os.path.exists(self.train_path): return
        print("📦 下载 WikiText-2...")
        url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip'
        try:
            r = requests.get(url, stream=True, timeout=30)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.data_dir)
            base = os.path.join(self.data_dir, 'wikitext-2-raw')
            os.rename(os.path.join(base, 'wiki.train.raw'), self.train_path)
            os.rename(os.path.join(base, 'wiki.valid.raw'), self.val_path)
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            sys.exit(1)

    def next_batch(self):
        # 循环读取流
        if self.current_idx + self.block_size + 1 > self.num_batches:
            self.current_idx = 0
        x = self.data[:, self.current_idx : self.current_idx + self.block_size]
        y = self.data[:, self.current_idx+1 : self.current_idx + self.block_size + 1]
        self.current_idx += self.block_size
        return x, y
    
    def decode(self, l):
        # 解码：直接用 chr() 还原 Unicode 字符
        # 过滤掉 0 (UNK) 以免打印乱码
        return ''.join([chr(i) if i > 0 else '' for i in l])

# ==========================================
# 2. 核心组件 (H2Q 架构 - 保持不变)
# ==========================================
class WaveStructureBank(nn.Module):
    def __init__(self, num_blocks, rank):
        super().__init__()
        self.sub_blocks = num_blocks // 4 
        self.rank = rank
        self.factors_A = nn.Parameter(torch.zeros(
            rank, 4, self.sub_blocks, self.sub_blocks, 
            device=device_structure, dtype=torch.float32
        ))
        with torch.no_grad():
            for r in range(rank):
                comps = torch.randn(4, self.sub_blocks, self.sub_blocks, device=device_structure)
                for c in range(4): nn.init.orthogonal_(comps[c])
                scale = (r + 1) ** -0.5
                self.factors_A.data[r] = comps * scale
    def get_factors(self): return self.factors_A

class BalancedHamiltonLayer(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, rank):
        super().__init__()
        self.dim = dim
        self.factor_size = factor_size
        self.structure_bank = structure_bank
        self.rank = rank
        self.factors_B = nn.Parameter(torch.zeros(rank, factor_size, factor_size, device=device_compute))
        self.bias = nn.Parameter(torch.zeros(dim, device=device_compute))
        with torch.no_grad():
            for r in range(rank):
                b = torch.randn(factor_size, factor_size, device=device_compute)
                nn.init.orthogonal_(b)
                scale = (r + 1) ** -0.5
                self.factors_B.data[r] = b * scale

    def _construct_hamilton_matrix_batch(self, A_comps_stack):
        r, i, j, k = A_comps_stack[:, 0], A_comps_stack[:, 1], A_comps_stack[:, 2], A_comps_stack[:, 3]
        row0 = torch.cat([r, -i, -j, -k], dim=2)
        row1 = torch.cat([i, r, -k, j], dim=2)
        row2 = torch.cat([j, k, r, -i], dim=2)
        row3 = torch.cat([k, -j, i, r], dim=2)
        return torch.cat([row0, row1, row2, row3], dim=1)

    def forward(self, x):
        B_batch, T, D = x.shape
        x_flat = x.view(-1, 4 * self.structure_bank.sub_blocks, self.factor_size)
        active_A = self.structure_bank.get_factors()
        A_stack = active_A.to(dtype=x.dtype)
        B_stack = self.factors_B.to(dtype=x.dtype)
        H_stack = self._construct_hamilton_matrix_batch(A_stack)
        wave_mod = torch.einsum('nsi, rji -> rnsj', x_flat, B_stack)
        wave_out = torch.einsum('rnsj, rks -> nkj', wave_mod, H_stack)
        return wave_out.reshape(B_batch, T, D) + self.bias

    def orthogonality_loss(self):
        loss = torch.tensor(0.0, device=device_compute)
        for p in self.factors_B:
            p_f32 = p.float()
            loss = loss + torch.norm(torch.mm(p_f32.t(), p_f32) - torch.eye(p.shape[1], device=device_compute))
        return loss

class QuaternionAttention(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, rank, num_heads=8):
        super().__init__()
        assert dim % 4 == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.quat_dim = self.head_dim // 4 
        self.q_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, rank)
        self.k_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, rank)
        self.v_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, rank)
        self.o_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, rank)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.quat_dim, 4)
        k = self.k_proj(x).view(B, T, self.num_heads, self.quat_dim, 4)
        v = self.v_proj(x).view(B, T, self.num_heads, self.quat_dim, 4)
        
        q_flat = q.reshape(B, self.num_heads, T, -1)
        k_flat = k.reshape(B, self.num_heads, T, -1)
        
        attn_score = (q_flat @ k_flat.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=device_compute) * float('-inf'), diagonal=1)
        attn_probs = F.softmax(attn_score + mask, dim=-1)
        
        y_flat = attn_probs @ v.reshape(B, self.num_heads, T, -1)
        y = y_flat.reshape(B, self.num_heads, T, self.quat_dim, 4)
        
        r, i, j, k_ = y.unbind(dim=-1)
        norm = torch.sqrt(r*r + i*i + j*j + k_*k_ + 1e-6)
        y = y / norm.unsqueeze(-1) * r.unsqueeze(-1)
        y = y.reshape(B, T, C)
        return self.o_proj(y)

    def orthogonality_loss(self):
        return (self.q_proj.orthogonality_loss() + self.k_proj.orthogonality_loss() + 
                self.v_proj.orthogonality_loss() + self.o_proj.orthogonality_loss())

class HolographicReversibleBlock(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, rank):
        super().__init__()
        self.half_dim = dim // 2
        self.norm1 = nn.LayerNorm(self.half_dim)
        self.attn = QuaternionAttention(self.half_dim, factor_size, structure_bank, rank, num_heads=8)
        self.norm2 = nn.LayerNorm(self.half_dim)
        self.ffn_1 = BalancedHamiltonLayer(self.half_dim, factor_size, structure_bank, rank)
        self.act = nn.GELU()
        self.ffn_2 = BalancedHamiltonLayer(self.half_dim, factor_size, structure_bank, rank)

    def f_block(self, x): return self.attn(self.norm1(x))
    def g_block(self, x): return self.ffn_2(self.act(self.ffn_1(self.norm2(x))))

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1 + checkpoint.checkpoint(self.f_block, x2, use_reentrant=False)
        y2 = x2 + checkpoint.checkpoint(self.g_block, y1, use_reentrant=False)
        return torch.cat([y1, y2], dim=-1)

    def get_ortho_loss(self):
        return self.attn.orthogonality_loss() + self.ffn_1.orthogonality_loss() + self.ffn_2.orthogonality_loss()

class H2Q_Transformer_Stream(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        half_dim = config['dim'] // 2
        num_blocks_half = half_dim // config['factor_size']
        self.structure_bank = WaveStructureBank(num_blocks_half, config['fixed_rank'])
        self.token_emb = nn.Embedding(vocab_size, config['dim'])
        self.pos_emb = nn.Parameter(torch.randn(1, config['seq_len'], config['dim']) * 0.02)
        self.drop = nn.Dropout(config['dropout_rate'])
        self.layers = nn.ModuleList([
            HolographicReversibleBlock(config['dim'], config['factor_size'], self.structure_bank, config['fixed_rank'])
            for _ in range(config['depth'])
        ])
        self.head = nn.Linear(config['dim'], vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        x = self.drop(self.token_emb(x) + self.pos_emb[:, :T, :])
        ortho_loss = torch.tensor(0.0, device=device_compute)
        for layer in self.layers:
            x = layer(x) 
            ortho_loss = ortho_loss + layer.get_ortho_loss()
        wave_energy = x.norm(dim=-1).mean()
        logits = self.head(x)
        loss = None
        if targets is not None:
            # 兼容 Stream reshape
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce_loss + self.config['axiom_lambda'] * ortho_loss * 0.01
        return logits, loss, wave_energy

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['seq_len']:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 3. 训练循环
# ==========================================
def format_log(step, train_loss, val_loss, grad_norm, energy, dt):
    diff = train_loss - val_loss
    vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    c_reset, c_diff, c_mem = "\033[0m", "\033[92m" if diff < 0 else "\033[91m", "\033[96m"
    return (f"Step {step:6d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Diff: {c_diff}{diff:+.4f}{c_reset} | Grad: {grad_norm:.2f} | "
            f"Energy: {energy:.2f} | VRAM: {c_mem}{vram:.2f}GB{c_reset} | Time: {dt:.1f}ms")

def train_h2q_micro():
    # 🔥 不需要手动传递词表了，因为大家都遵守 Unicode 标准
    train_loader = SequentialLoader(CONFIG['seq_len'], CONFIG['batch_size'], CONFIG['data_dir'], 'train')
    val_loader = SequentialLoader(CONFIG['seq_len'], CONFIG['batch_size'], CONFIG['data_dir'], 'val')
    
    # 词表大小统一为 256
    model = H2Q_Transformer_Stream(256, CONFIG)
    model.to(device_compute)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 H2Q-Micro 模型参数量: {total_params/1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    
    loss_history = []
    best_val_loss = float('inf')
    start_iter = 0

    if os.path.exists(CONFIG['checkpoint_path']):
        print(f"🔄 恢复存档: {CONFIG['checkpoint_path']}")
        ckpt = torch.load(CONFIG['checkpoint_path'], map_location=device_compute, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter']
        loss_history = ckpt['loss_history']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

    print("🚀 启动 H2Q-MicroStream 训练 (Unicode Standard)...")
    model.train()
    optimizer.zero_grad()
    
    try:
        for iter_num in range(start_iter, CONFIG['total_iters']):
            t0 = time.time()
            
            xb, yb = train_loader.next_batch()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss, energy = model(xb, yb)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            loss_history.append(loss.item())
            dt = (time.time() - t0) * 1000
            
            if iter_num % CONFIG['eval_interval'] == 0:
                vx, vy = val_loader.next_batch()
                model.eval()
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        _, val_loss, _ = model(vx, vy)
                model.train()
                print(format_log(iter_num, loss.item(), val_loss.item(), total_norm, energy.item(), dt))
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save({'model': model.state_dict(), 'config': CONFIG}, CONFIG['best_model_path'])

            if iter_num > 0 and iter_num % 1000 == 0:
                torch.save({'iter': iter_num, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss_history': loss_history, 'best_val_loss': best_val_loss}, CONFIG['checkpoint_path'])
                print("\n📜 [Unicode Generation]:")
                model.eval()
                with torch.no_grad():
                    # 初始种子设为 Unicode 空格 (32)
                    ctx = torch.tensor([[32]], dtype=torch.long, device=device_compute)
                    out = model.generate(ctx, 150)
                    print(train_loader.decode(out[0].tolist()))
                model.train()
                print("-" * 50)

    except KeyboardInterrupt:
        print("\n🛑 保存并退出...")
        torch.save({'iter': iter_num, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss_history': loss_history, 'best_val_loss': best_val_loss}, CONFIG['checkpoint_path'])

if __name__ == "__main__":
    train_h2q_micro()