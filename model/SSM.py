import torch
import torch.nn as nn
import torch.nn.functional as F

"""动态滤波模块"""

#对角矩阵
class DynamicSSM(nn.Module):
    def __init__(self, frequency_size, scale):                                      #frequency_size = d
        super().__init__()
        def complex_init(shape):
            magnitude = torch.rand(shape) * scale                                   #幅度
            phase = torch.rand(shape) * 2 *torch.pi                                 #相位
            return torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        # 初始化对角矩阵参数（存储对角线元素）
        self.A = nn.Parameter(complex_init(frequency_size))                         #[d]                   
        self.B = nn.Parameter(complex_init(frequency_size))                         #[d]
        self.C = nn.Parameter(complex_init(frequency_size))                         #[d]

    def forward(self, x_dynamic):
        #[B*f, K, d]
        n_freq = x_dynamic.shape[1]                                 #K
        angular_freq = 2j * torch.pi * torch.arange(n_freq, device=x_dynamic.device) / n_freq   # 角频率：2jπ * k/N, k为频率索引 [k]
        
        e_term = torch.exp(-angular_freq).view(-1, 1)               # [k, 1]
        # Hadamard积计算（逐元素操作）
        denominator = 1 - self.A * e_term                           # [K, d]
        H = (self.C * self.B) / denominator                         # [K, d]
        x_out = x_dynamic * H.unsqueeze(0)                          # [B*f, K, d] 
        return x_out                                                # [B*f, K, d]

#低秩分解
class DynamicSSM_LowRank(nn.Module):
    def __init__(self, frequency_size, scale, rank=4):
        super().__init__()
        self.frequency = frequency_size
        self.rank = rank

        # 定义实部和虚部分解函数
        def real_imag_param(shape):
            return (
                nn.Parameter(torch.randn(shape) * scale),  # 实部
                nn.Parameter(torch.randn(shape) * scale)   # 虚部
            )
        
        # 初始化低秩因子,实部虚部分离 [d, r]
        self.U_A_real, self.U_A_imag = real_imag_param((frequency_size, rank))      
        self.V_A_real, self.V_A_imag = real_imag_param((frequency_size, rank))
        self.U_B_real, self.U_B_imag = real_imag_param((frequency_size, rank))
        self.V_B_real, self.V_B_imag = real_imag_param((frequency_size, rank))
        self.U_C_real, self.U_C_imag = real_imag_param((frequency_size, rank))
        self.V_C_real, self.V_C_imag = real_imag_param((frequency_size, rank))

    def forward(self, x_dynamic):
        #[B*f, K, d]
        B, n_freq, D = x_dynamic.shape
        angular_freq = 2j * torch.pi * torch.arange(n_freq, device=x_dynamic.device) / n_freq       # 角频率 [n_freq]
        e_term = torch.exp(-angular_freq).view(-1, 1, 1)                                            # [n_freq, 1, 1]

        # 重建复数矩阵（避免直接使用 torch.cfloat 参数）
        def build_complex(real, imag):
            return torch.complex(real, imag)
        
        U_A = build_complex(self.U_A_real, self.U_A_imag)           #[d, r]
        V_A = build_complex(self.V_A_real, self.V_A_imag)           #[d, r]
        A = U_A @ V_A.conj().T                                      #[d, d]

        U_B = build_complex(self.U_B_real, self.U_B_imag)           #[d, r]           
        V_B = build_complex(self.V_B_real, self.V_B_imag)           #[d, r]
        B = U_B @ V_B.conj().T                                      #[d, d]

        U_C = build_complex(self.U_C_real, self.U_C_imag)           #[d, r]
        V_C = build_complex(self.V_C_real, self.V_C_imag)           #[d, r]
        C = U_C @ V_C.conj().T                                      #[d, d]

        # 计算传递函数 H = (C @ B) / (1 - A * e_term)
        denominator = 1 - A * e_term                                # [K, d, d]
        H = (C @ B) / denominator                                   # [K, d, d]

        x_out = torch.einsum('bfi,fij->bfj', x_dynamic, H)          # [B*f, K, d]          

        return x_out
    
###块对角矩阵：embed_size/number_frequency=32,64,128
class DynamicSSM_BlockDiagonal(nn.Module):
    def __init__(self, frequency_size, scale, block_size = 32):
        super().__init__()
        assert frequency_size % block_size == 0, "frequency_size must be divisible by block_size"
        # # 打印参数值
        # print(f"frequency_size = {frequency_size}, block_size = {block_size}")

        self.frequency = frequency_size
        self.block_size = block_size
        self.num_blocks = frequency_size // block_size

        # 初始化块对角参数 (复数矩阵)
        self.A_blocks = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size, dtype=torch.cfloat) * scale)
            for _ in range(self.num_blocks)
        ])
        self.B_blocks = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size, dtype=torch.cfloat) * scale)
            for _ in range(self.num_blocks)
        ])
        self.C_blocks = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size, dtype=torch.cfloat) * scale)
            for _ in range(self.num_blocks)
        ])

    def forward(self, x_dynamic):
        #[B*f, K, d]
        B, n_freq, D = x_dynamic.shape                  
        angular_freq = 2j * torch.pi * torch.arange(n_freq, device=x_dynamic.device) / n_freq   #[k]

        e_term = torch.exp(-angular_freq).view(-1, 1, 1)                                        #  [K, 1, 1]

        H = torch.zeros(B, n_freq, D, dtype=torch.cfloat, device=x_dynamic.device)

        for i in range(self.num_blocks):
            start = i * self.block_size
            end = (i+1) * self.block_size

            A_block = self.A_blocks[i]      #[block_size, block_size]
            B_block = self.B_blocks[i]
            C_block = self.C_blocks[i]

            denominator = torch.eye(self.block_size, device=x_dynamic.device) - A_block[None,:,:] * e_term      #[k, block_size, block_size]
            H_block = torch.linalg.solve(denominator, C_block @ B_block)                                        #[k, block_size, block_size]

            x_block = x_dynamic[:, :, start:end]                                                                # [B*f, k, block_size]
            filtered_block = torch.einsum('bfi,fij->bfj', x_block, H_block)                                     # [B*f, k, block_size]      
            H[:, :, start:end] = filtered_block                                                                 # [B*f, k, d]
            x_out = H

        return x_out                             


#稠密矩阵
class DynamicSSM_Dense(nn.Module):
    def __init__(self, frequency_size, scale):
        super().__init__()
        self.frequency = frequency_size     

        def complex_init(shape):
            magnitude = torch.rand(shape) * scale                                   #幅度
            phase = torch.rand(shape) * 2 *torch.pi                                 #相位
            return torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        # 初始化稠密复数矩阵 [d, d]
        self.A = nn.Parameter(complex_init((frequency_size, frequency_size)))                                            
        self.B = nn.Parameter(complex_init((frequency_size, frequency_size)))                         
        self.C = nn.Parameter(complex_init((frequency_size, frequency_size)))    
                

    def forward(self, x_dynamic):
        #[B*f, K, d]
        n_freq = x_dynamic.shape[1]                                 #K

        angular_freq = 2j * torch.pi * torch.arange(n_freq, device=x_dynamic.device) / n_freq           # 角频率 [K]
        e_term = torch.exp(-angular_freq).view(-1, 1, 1)                                                # [K, 1, 1]

        I = torch.eye(self.frequency, dtype=torch.complex64, device=x_dynamic.device)                   # [d, d]
        denominator = I.unsqueeze(0) - self.A.unsqueeze(0) * e_term                                     # [K, d, d]

        # 矩阵求逆并计算 H = C @ inv(denominator) @ B
        inv_denominator = torch.linalg.inv(denominator)                                                 # [K, d, d]
        H = torch.einsum("ij,kjl->kil", self.C, inv_denominator)                                        # [K, d, d]
        H = torch.einsum("kil,lj->kij", H, self.B)                                                      # [k, d, d]

        x_out = torch.einsum("bki,kij->bkj", x_dynamic.to(torch.complex64), H)                          #[B*f, K, d]                           
        return x_out  
    
                                                                  