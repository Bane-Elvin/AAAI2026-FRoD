import torch
from peft import PeftModel
from transformers import AutoModel
from collections import defaultdict
import numpy.linalg as la
import numpy as np
import torch.nn.functional as F

base_model_name_1 = "/your/path/to/models/vit-base-patch16-224"
base_model_name_2 = "/your/path/to/models/vit-base-cars-Full-FT-final"
# lora_model_path = "./llama-3.1-8B-gsm8k_5k-lorara-finetuned"

print(base_model_name_1)
# torch.manual_seed(20250801)

base_model_1 = AutoModel.from_pretrained(base_model_name_1)
ft_model = AutoModel.from_pretrained(base_model_name_2)
# peft_model = PeftModel.from_pretrained(base_model, lora_model_path)
# print(base_model)

categories = ['key', 'query', 'value']

original_weights_1 = defaultdict(dict)
original_weights_2 = defaultdict(dict)
base_state_dict_1 = base_model_1.state_dict()
base_state_dict_2 = ft_model.state_dict()
for name, weight in base_state_dict_1.items():
    parts = name.split('.')
    if 'encoder' in parts and 'weight' in parts and len(parts) >= 6:
        layer_idx = int(parts[2])
        original_weights_1[layer_idx][parts[5]] = weight.cpu().detach()
for name, weight in base_state_dict_2.items():
    parts = name.split('.')
    if 'encoder' in parts and 'weight' in parts and len(parts) >= 6:
        layer_idx = int(parts[2])
        original_weights_2[layer_idx][parts[5]] = weight.cpu().detach()

# print(original_weights[1]['key'].shape)

device = torch.device('cpu')
def zero_result():
    error_dict = defaultdict(dict)
    for layer_idx in original_weights_1:
        for category in categories:
            W = original_weights_1[layer_idx][category]
            # W2 = original_weights_2[layer_idx][category]
            # W = W2 - W1
            dim_out, dim_in = W.shape
            zero_matrix = torch.zeros(dim_out, dim_out)
            error = torch.norm(W - zero_matrix, p='fro')
            error_dict[category][layer_idx] = error.item()
            print(f"{layer_idx} {category} recontribution error: {error_dict[category][layer_idx]}")

def random_SVD_result(num_trials=1):

    from collections import defaultdict
    error_dict = defaultdict(dict)

    for layer_idx in original_weights_1:
        for category in categories:
            W1 = original_weights_1[layer_idx][category]
            W2 = original_weights_2[layer_idx][category]
            W = W2 - W1
            dim_out, dim_in = W.shape

            best_error = float('inf')

            for _ in range(num_trials):
                rand_U, _ = torch.linalg.qr(torch.randn(dim_out, dim_out))
                rand_V, _ = torch.linalg.qr(torch.randn(dim_in, dim_in))

                Q = rand_U.T @ W @ rand_V
                S_diag = torch.diag(torch.diagonal(Q))

                W_approx = rand_U @ S_diag @ rand_V.T

                error = torch.norm(W - W_approx, p='fro')

                if error < best_error:
                    best_error = error.item()

            error_dict[category][layer_idx] = best_error
            print(f"{layer_idx} {category} recontribution error: {error_dict[category][layer_idx]}")


def Sparse_SVD_result(num_trials=1, topk=768*8):

    from collections import defaultdict
    error_dict = defaultdict(dict)

    for layer_idx in original_weights_1:
        for category in categories:
            W1 = original_weights_1[layer_idx][category]
            W2 = original_weights_2[layer_idx][category]
            W = W2 - W1
            dim_out, dim_in = W.shape

            best_error = float('inf')

            for _ in range(num_trials):
                rand_U, _ = torch.linalg.qr(torch.randn(dim_out, dim_out))
                rand_V, _ = torch.linalg.qr(torch.randn(dim_in, dim_in))

                Q = rand_U.T @ W @ rand_V

                Q_flat = Q.view(-1)
                topk = min(topk, Q_flat.numel())
                topk_vals, topk_indices = torch.topk(torch.abs(Q_flat), topk, largest=True, sorted=False)

                Q_sparse = torch.zeros_like(Q_flat)
                Q_sparse[topk_indices] = Q_flat[topk_indices]
                Q_sparse = Q_sparse.view_as(Q)

                W_approx = rand_U @ Q_sparse @ rand_V.T
                error = torch.norm(W - W_approx, p='fro')

                if error < best_error:
                    best_error = error.item()

            error_dict[category][layer_idx] = best_error
            print(f"{layer_idx} {category} recontribution error: {error_dict[category][layer_idx]}")


def randLoRA_result(num_trials=1,r = 8):

    from collections import defaultdict
    error_dict = defaultdict(dict)


    for layer_idx in original_weights_1:
        for category in categories:
            W1 = original_weights_1[layer_idx][category]
            W2 = original_weights_2[layer_idx][category]
            W = W2 - W1
            dim_out, dim_in = W.shape

            d = min(dim_out, dim_in)
            if d % r != 0:
                continue

            best_error = float('inf')

            for _ in range(num_trials):
                rand_U, _ = torch.linalg.qr(torch.randn(dim_out, dim_out))
                rand_V, _ = torch.linalg.qr(torch.randn(dim_in, dim_in))

                Q = rand_U.T @ W @ rand_V
                Q_block = torch.zeros_like(Q)

                for i in range(0, d, r):
                    Q_block[i:i+r, i:i+r] = Q[i:i+r, i:i+r]

                W_approx = rand_U @ Q_block @ rand_V.T
                error = torch.norm(W - W_approx, p='fro')

                if error < best_error:
                    best_error = error.item()

            error_dict[category][layer_idx] = best_error
            print(f"{layer_idx} {category} recontribution error: {error_dict[category][layer_idx]}")

def lora_result():
    error_dict = defaultdict(dict)
    for layer_idx in original_weights_1:
        for category in categories:

            W = original_weights_1[layer_idx][category]
            # W2 = original_weights_2[layer_idx][category]
            # W = W2 - W1
            dim_out, dim_in = W.shape

            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            k = 8
            S_truncated = torch.zeros_like(S)
            S_truncated[:k] = S[:k]
            S_mat = torch.diag(S_truncated)
            W_k = U @ S_mat @ Vh


            error = torch.norm(W - W_k, p='fro')
            error_dict[category][layer_idx] = error.item()

            print(f"{layer_idx} {category} recontribution error: {error_dict[category][layer_idx]}")

def HO_GSVD_result():
    error_dict = defaultdict(dict)
    c_dict = defaultdict(dict)
    S_ij_dict = defaultdict(dict)
    for category in categories:
        c_dict[category] = []
        S_ij_dict[category] = []
        for layer_idx in original_weights_1:
            W1 = original_weights_1[layer_idx][category]
            W2 = original_weights_2[layer_idx][category]
            W = W2 - W1
            c_dict[category].append(W)
        for i in range(len(c_dict[category])):
            for j in range(i + 1, len(c_dict[category])):
                A_i, A_j = c_dict[category][i], c_dict[category][j]
                S_ij = 0.5 * (A_i @ la.inv(A_j) + A_j @ la.inv(A_i))
                S_ij_dict[category].append(S_ij)

        S = sum(S_ij_dict[category]) / len(S_ij_dict[category])
        eigvals, V = la.eig(S)
        V = np.real_if_close(V)
        # V = V.real
        eigvals = np.real_if_close(eigvals)
        V_inv_T = la.inv(V).T
        Bi_list = [D @ V_inv_T for D in c_dict[category]]

        Ui_list = []
        Si_list = []
        for i, Bi in enumerate(Bi_list):
            si = la.norm(Bi, axis=0)
            Si = np.diag(si)
            Ui = Bi / si
            Ui_list.append(Ui)
            Si_list.append(Si)
            for k in range(Ui.shape[1]):
                for l in range(k + 1, Ui.shape[1]):
                    dot_val = np.dot(Ui[:, k], Ui[:, l])
                    print(f"u_{k} · u_{l} = {dot_val:.3e}")


            D_hat = Ui @ Si @ V.T
            error = la.norm(c_dict[category][i] - D_hat)
            error_dict[category][i] = error
            print(f"{layer_idx} {category} recontribution error: {error_dict[category][layer_idx]}")

def HO_GSVD_result_with_regularization(pi=1e-3):
    """
    HO-GSVD with Tikhonov-like regularization term pi*A^T*A.
    """
    from collections import defaultdict

    error_dict = defaultdict(dict)
    c_dict = defaultdict(list)
    S_ij_dict = defaultdict(list)

    for category in categories:
        A_list = []
        for layer_idx in original_weights_1:
            W1 = original_weights_1[layer_idx][category]
            W2 = original_weights_2[layer_idx][category]
            # A_i = (W2 - W1).numpy()
            A_i = W1.numpy()
            A_list.append(A_i.T)
            c_dict[category].append(A_i)

        # calculate A = [A1; A2; ...; AN]
        A_stack = np.vstack(A_list)
        A_global = A_stack.T @ A_stack

        #  S_ij with regularization
        for i in range(len(c_dict[category])):
            for j in range(i + 1, len(c_dict[category])):
                Ai, Aj = c_dict[category][i], c_dict[category][j]

                Di_pi = Ai @ Ai.T + pi * A_global
                Dj_pi = Aj @ Aj.T + pi * A_global

                Di_pi_inv = la.inv(Di_pi)
                Dj_pi_inv = la.inv(Dj_pi)

                S_ij = 0.5 * (Di_pi @ Dj_pi_inv + Dj_pi @ Di_pi_inv)
                S_ij_dict[category].append(S_ij)

        S = sum(S_ij_dict[category]) / len(S_ij_dict[category])

        eigvals, V = la.eig(S)
        V = np.real_if_close(V)
        eigvals = np.real_if_close(eigvals)
        V_inv_T = la.inv(V).T
        print(f"Orthogonality check: {V.T @ V }")


        Bi_list = [D @ V_inv_T for D in c_dict[category]]

        for i, Bi in enumerate(Bi_list):
            si = la.norm(Bi, axis=0)
            Si = np.diag(si)
            Ui = Bi / si

            D_hat = Ui @ Si @ V.T
            Ui_T_Ui = Ui.T @ Ui
            print(f"{i}th {category} Ui ORG:\n", Ui_T_Ui - np.eye(Ui_T_Ui.shape[0]))
            orth_error = np.linalg.norm(Ui_T_Ui - np.eye(Ui_T_Ui.shape[0]), ord='fro')
            print(f"Ui Frobenius : {orth_error:}")
            error = la.norm(c_dict[category][i] - D_hat)
            error_dict[category][i] = error
            print(f"{layer_idx} {category} recontribution error: {error_dict[category][layer_idx]}")


def HO_CSD_result_with_regularization(pi=1e-3):

    import numpy as np
    from numpy.linalg import qr, inv, eigh, norm
    from collections import defaultdict

    error_dict = defaultdict(dict)
    Q_dict = defaultdict(list)
    A_list = []

    for category in categories:
        A_list = []
        Q_dict[category] = []
        for layer_idx in original_weights_1:
            W1 = original_weights_1[layer_idx][category]
            # W2 = original_weights_2[layer_idx][category]
            # A_i = (W2 - W1).numpy()

            # low rank
            # U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
            # k = 8
            # S_truncated = torch.zeros_like(S)
            # S_truncated[:k] = S[:k]
            # S_mat = torch.diag(S_truncated)
            # W1 = U @ S_mat @ Vh


            A_i = W1.numpy()
            A_list.append(A_i)
            Q_dict[category].append(W1)

        # thin QR
        A_stack = np.vstack(A_list)
        Q, R = np.linalg.qr(A_stack)

        #get Qi
        Qi_list = []
        m = 0
        for A_i in A_list:
            mi = A_i.shape[0]
            Qi = Q[m: m + mi, :]
            Qi_list.append(Qi)
            m += mi

        # get T_pi 
        T_pi = np.zeros((R.shape[1], R.shape[1]))
        for Qi in Qi_list:
            # (Qi^T Qi + pi*I)^-1

            Qi_term = Qi.T @ Qi + pi * np.eye(Qi.shape[1])
            Qi_term_inv = inv(Qi_term)
            T_pi += Qi_term_inv
        T_pi /= len(Qi_list)

        # T_pi decomposition
        tau, Z = eigh(T_pi)
        Z = np.real_if_close(Z)


        # V = R^T Z 
        V = R.T @ Z
        # print(f"{category} V ORG:\n", V.T @ V)
        V = torch.from_numpy(V).to(device).float()

        for i, Qi in enumerate(Qi_list):
            Bi = Qi @ Z
            sigma_i = norm(Bi, axis=0)

            Ui = np.divide(Bi, sigma_i, where=sigma_i > 1e-8)


            Ui_T_Ui = Ui.T @ Ui
            # print(f"{i}th {category} Ui ORG:\n", Ui_T_Ui )
            orth_error = np.linalg.norm(Ui_T_Ui - np.eye(Ui_T_Ui.shape[0]), ord=2)
            # print(f"Ui error norm-2: {orth_error:}")




            # Ai_hat = Ui @ np.diag(sigma_i) @ V.T
            n = V.shape[0]

            h = torch.randn(1, n, device=device)

            U = torch.from_numpy(Ui).to(device).float()  # shape [m, n]
            l = torch.from_numpy(sigma_i).to(device).float()  # shape [n]
            L =torch.diag_embed(l)

            O_hat = F.linear(F.linear(F.linear(h, V.T), L), U)
            Ai_orig = Q_dict[category][i]
            # err_Ai = norm(Ai_orig - Ai_hat)
            O_orig = (Ai_orig @ h.T).T
            err_Ai = norm(O_orig - O_hat.detach().cpu().numpy())
            print(f"{layer_idx} {category} recontribution error: {error_dict[category][layer_idx]}")



if __name__ == '__main__':
    # Sparse_SVD_result()
    # randLoRA_result()
    # random_SVD_result()
    # lora_result()
    # HO_GSVD_result()
    # HO_GSVD_result_with_regularization()
    # zero_result()
    HO_CSD_result_with_regularization()


