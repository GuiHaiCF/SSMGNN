import torch
import torch.nn as nn
import torch.nn.functional as F

class StaticFGO(nn.Module):
    def __init__(self, embed_size, scale):
        super().__init__()

        magnitude = torch.rand(embed_size, embed_size) * scale                      #幅度
        phase = torch.rand(embed_size, embed_size) * 2 * torch.pi                   #相位
        self.W = nn.Parameter(torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase)))
        self.b = nn.Parameter(torch.randn(embed_size, dtype=torch.cfloat) * scale)

    def forward(self, x_static):
        #[B*f, K, D]
        static_out = torch.einsum('bfd,dd->bfd', x_static, self.W) + self.b         #[B*f, K, D]
        return static_out                   


class DynamicSSM(nn.Module):
    def __init__(self, frequency_size, scale):
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
        angular_freq = 2j * torch.pi * torch.arange(n_freq, device=x_dynamic.device) / n_freq   # 角频率：2jπ * k/N, k为频率索引 [n_freq]
        
        e_term = torch.exp(-angular_freq).view(-1, 1)               # 构造指数项 e^{-jω}，view(-1,1) 用于广播 [n_freq, 1]

        #  # 对A进行裁剪，确保幅度不超过1
        # A_abs = torch.abs(self.A)
        # self.A = self.A / torch.clamp_min(A_abs, 1.0)

        # Hadamard积计算（逐元素操作）
        denominator = 1 - self.A * e_term                           # 广播后维度 [K, d]
        H = (self.C * self.B) / denominator                         # [K, d]
        return x_dynamic * H.unsqueeze(0)                           # 输出维度保持 [B*f, K, d]
    
class HybridFGOLayer(nn.Module):
    def __init__(self, embed_size, number_frequency, frequency_size, scale=0.02, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.number_frequency = number_frequency
        self.frequency_size = frequency_size
        self.scale = scale
        self.sparsity_threshold=sparsity_threshold
        self.static_filter = StaticFGO(self.frequency_size, self.scale)
        self.dynamic_filter = DynamicSSM(self.frequency_size, self.scale)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_fft):
        #[B, K, D, 2]
        B, K, D, _ = x_fft.shape
        x_fft = x_fft.reshape(B, K, self.number_frequency, self.frequency_size, 2)              #[B, K, f, d, 2]
        x_fft = x_fft.permute(0,2,1,3,4)                                                        #[B, f, K, d, 2]
        x_fft = x_fft.reshape(B*self.number_frequency, K, self.frequency_size, 2)               #[B*f, K, d, 2]
        x_fft = torch.view_as_complex(x_fft)                                                    #[B*f, K, D]

        """静态滤波"""
        static_out = self.static_filter(x_fft)                                                  #[B*f, K, D]
        """动态滤波"""
        dynamic_out = self.dynamic_filter(x_fft)                                                #[B*f, K, d ]

        alpha = torch.sigmoid(self.alpha)
        combined = static_out + alpha * dynamic_out                                             #[B*f, K, d]

        combined = combined.reshape(B, self.number_frequency, K, self.frequency_size)           #[B, f, K, d ]
        combined = combined.permute(0,2,1,3)                                                    #[B, K, f, d ]
        combined = combined.reshape(B, K, D)                                                    #[B, K, D]

        out_real = F.relu(combined.real)                                                        #[B, K, D]
        out_imag = F.relu(combined.imag)                                                        #[B, K, D]
        out = torch.stack([out_real, out_imag], dim=-1)                                         #[B, K, D, 2]
        return F.softshrink(out, lambd=self.sparsity_threshold)                                 #[B, K, D, 2]




class SSMGNN(nn.Module):
    def __init__(self, pre_length, embed_size, feature_size, seq_length,
                 hidden_size, number_frequency, num_layers=3):
        super().__init__()
        self.pre_length = pre_length
        self.embed_size = embed_size
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.number_frequency = number_frequency
        self.num_layers = num_layers

        self.frequency_size = self.embed_size // self.number_frequency
        self.embeddings = nn.Parameter(torch.randn(1,self.embed_size))

        self.layers = nn.ModuleList([HybridFGOLayer(self.embed_size, self.number_frequency, self.frequency_size) for _ in range(self.num_layers)])

        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length,8))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')

    def tokenEmb(self, x):
        """输入嵌入"""
        x = x.unsqueeze(2)              
        y = self.embeddings
        return x * y
    
    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()                         #[B, L, N]->[B, N, L]
        B,N,L = x.shape
        x = x.reshape(B, -1)                                        #[B, N*L]
        x = self.tokenEmb(x)                                        #[B, N*L, D]

        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')              #[B, K, D] 
        x_fft = torch.stack([x_fft.real, x_fft.imag], dim=-1)       #[B, K, D, 2]

        bias = x_fft

        for layer in self.layers:
            x_fft = layer(x_fft) + bias                             #[B, K, D, 2]
            bias = x_fft

        x_fft = torch.view_as_complex(x_fft)                        #[B, K, D]
        x = torch.fft.irfft(x_fft, n=N*L, dim=1, norm='ortho')      #[B, N*L, D]
        x = x.reshape(B, N, L, self.embed_size)                     #[B, N, L, D]
        x = x.permute(0,1,3,2)                                      #[B, N, D, L]

        x = torch.matmul(x, self.embeddings_10)                     #[B, N, D, 8]
        x = x.reshape(B, N, -1)                                     #[B, N, D, 8]
        return self.fc(x)                                           #[B, N, pre_lenth]                                           