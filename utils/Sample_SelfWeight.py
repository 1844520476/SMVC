import torch
import torch.nn.functional as F
from torch.nn import Linear


class SelfAttentionMechanism:
    def __init__(self, feature_dim):
        self.query_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.key_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.value_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.out_layer = Linear(feature_dim, feature_dim, bias=False).cuda()
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32)).cuda()

    def compute_attention_weights(self, z_all, zs):
        batch_size = z_all.size(0)
        view_count = len(zs)

        # 线性变换获取查询、键和值
        Q = self.query_layer(z_all.cuda())  # [batch_size, feature_dim]
        K = torch.stack([self.key_layer(z.cuda()) for z in zs], dim=1)  # [batch_size, view_count, feature_dim]
        V = torch.stack([self.value_layer(z.cuda()) for z in zs], dim=1)  # [batch_size, view_count, feature_dim]

        # 检查形状是否正确
        assert Q.shape == (
        batch_size, Q.size(-1)), f"Query shape mismatch: expected {(batch_size, Q.size(-1))}, got {Q.shape}"
        assert K.shape == (batch_size, view_count, K.size(
            -1)), f"Key shape mismatch: expected {(batch_size, view_count, K.size(-1))}, got {K.shape}"
        assert V.shape == (batch_size, view_count, V.size(
            -1)), f"Value shape mismatch: expected {(batch_size, view_count, V.size(-1))}, got {V.shape}"

        # 计算点积得分
        scores = torch.einsum('bf,bvf->bv', Q, K) / self.scale  # [batch_size, view_count]

        # 对每个样本的视图评分应用softmax以获取权重
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, view_count]

        # 计算注意力输出
        attention_output = torch.einsum('bv,bvf->bf', attention_weights, V)  # [batch_size, feature_dim]

        # 通过最后的线性变换
        output = self.out_layer(attention_output)

        return output, attention_weights


# 测试代码
if __name__ == "__main__":
    batch_size = 4
    feature_dim = 5
    view_count = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_all = torch.randn(batch_size, feature_dim).to(device)
    zs = [torch.randn(batch_size, feature_dim).to(device) for _ in range(view_count)]

    # 创建一个部分样本部分视图为全零的测试
    zs[0][0] = torch.zeros(feature_dim).to(device)
    zs[2][0] = torch.zeros(feature_dim).to(device)

    zs[2][1] = torch.zeros(feature_dim).to(device)  # 第三个视图中第二个样本的全零
    zs[1][2] = torch.zeros(feature_dim).to(device)
    zs[2][3] = torch.zeros(feature_dim).to(device)

    attention_mechanism = SelfAttentionMechanism(feature_dim)
    output, weights = attention_mechanism.compute_attention_weights(z_all, zs)
    print("Attention Weights:\n", weights)
    print("Output:\n", output)
