import torch
import torch.nn as nn
import torch.nn.functional as F # Add this import

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        # self.similarity_f: Will use F.cosine_similarity or manual mm

    def _create_mask(self, current_batch_size_on_gpu):
        """
        Creates a mask for excluding positive pairs and self-correlations
        based on the actual batch size processed by the current GPU.
        """
        N_slice = 2 * current_batch_size_on_gpu
        mask = torch.ones((N_slice, N_slice), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(current_batch_size_on_gpu):
            mask[i, current_batch_size_on_gpu + i] = 0
            mask[current_batch_size_on_gpu + i, i] = 0
        return mask

    def forward(self, out_i, out_j):
        """
        Args:
            out_i: Embeddings of the first view for the current GPU's slice [B_slice, D_projected]
            out_j: Embeddings of the second view for the current GPU's slice [B_slice, D_projected]
        """
        b_slice = out_i.shape[0]  # Actual batch size for this GPU's slice
        n_slice = 2 * b_slice

        if b_slice == 0: # Handle empty batch slice
            return torch.tensor(0.0, device=out_i.device, requires_grad=True)

        # Normalize the representations
        out_i_norm = F.normalize(out_i, dim=1)
        out_j_norm = F.normalize(out_j, dim=1)

        # Concatenate representations: [2*B_slice, D_projected]
        representations = torch.cat([out_i_norm, out_j_norm], dim=0)

        # Compute similarity matrix: [2*B_slice, 2*B_slice]
        # Using matrix multiplication for cosine similarity:
        sim_matrix = torch.mm(representations, representations.t().contiguous()) / self.temperature

        # Positive samples (diagonal elements after aligning views)
        # sim_i_j corresponds to sim(out_i[k], out_j[k])
        # sim_j_i corresponds to sim(out_j[k], out_i[k])
        sim_i_j = torch.diag(sim_matrix, b_slice)  # Extracts upper-off-diagonal
        sim_j_i = torch.diag(sim_matrix, -b_slice) # Extracts lower-off-diagonal

        # Concatenate positive similarities: [2*B_slice] -> [2*B_slice, 1]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n_slice, 1)

        # Create mask dynamically for the current slice size
        mask_slice = self._create_mask(b_slice).to(sim_matrix.device)

        # Negative samples: all elements not masked
        negative_samples = sim_matrix[mask_slice].reshape(n_slice, -1)

        # Logits for CrossEntropyLoss: [2*B_slice, N_slice-1]
        # (first column is positive, rest are negatives)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        # Labels: positive samples are at index 0 for each row
        labels = torch.zeros(n_slice, device=logits.device, dtype=torch.long)

        loss = self.criterion(logits, labels)
        loss = loss / n_slice  # Normalize by the actual number of samples in this slice
        return loss
