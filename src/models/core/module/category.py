import numpy as np
import torch
import torch.nn as nn



class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(
        self,
        input_dim: int,
        cat_idxs: list,
        cat_dims: list,
        cat_emb_dims: int | list,
        group_matrix: torch.Tensor,
    ):
        super().__init__()

        # (1) 임베딩을 쓰지 않는 경우: 원본 group_matrix 그대로 버퍼 등록
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            self.register_buffer(
                "embedding_group_matrix",
                group_matrix.to(torch.float32)
            )
            return

        # (2) 임베딩을 쓰는 경우
        self.skip_embedding = False

        # 임베딩 차원 계산
        self.post_embed_dim = int(input_dim + np.sum(cat_emb_dims) - len(cat_emb_dims))

        # 임베딩 레이어들 생성
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims)]
        )

        # 연속형/범주형 인덱스 마스크 생성 → 버퍼로 등록(디바이스 자동 이동용)
        continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        continuous_idx[cat_idxs] = 0
        self.register_buffer("continuous_idx", continuous_idx)

        # (3) 임베딩 적용 후의 그룹 행렬 구성
        n_groups = group_matrix.shape[0]
        embedding_group_matrix = torch.empty(
            (n_groups, self.post_embed_dim),
            device=group_matrix.device,
            dtype=torch.float32,
        )

        for g in range(n_groups):
            post_emb_idx = 0
            cat_feat_counter = 0
            for init_feat_idx in range(input_dim):
                if self.continuous_idx[init_feat_idx]:
                    # 연속형: 차원 증가 없음 → 그대로 한 칸
                    embedding_group_matrix[g, post_emb_idx] = group_matrix[g, init_feat_idx]
                    post_emb_idx += 1
                else:
                    # 범주형: emb_dim만큼 확장 → 값은 균등 분배
                    n_embeddings = int(cat_emb_dims[cat_feat_counter])
                    embedding_group_matrix[g, post_emb_idx:post_emb_idx + n_embeddings] = (
                        group_matrix[g, init_feat_idx] / n_embeddings
                    )
                    post_emb_idx += n_embeddings
                    cat_feat_counter += 1

        # (4) 완성된 행렬을 버퍼로 등록
        self.register_buffer("embedding_group_matrix", embedding_group_matrix)


    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings
