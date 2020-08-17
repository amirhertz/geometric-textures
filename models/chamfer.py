'''
Based on SO-Net implementation
https://github.com/lijx10/SO-Net/blob/master/models/losses.py
'''
from custom_types import *
import faiss


def robust_norm(var):
    result = ((var**2).sum(dim=2) + 1e-8).sqrt()
    return result


class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()
        #
        self.device = device
        self.gpu_id = device.index

        # we need only a StandardGpuResources per GPU
        self.res = faiss.StandardGpuResources()
        # self.res.setTempMemoryFraction(0.1)
        self.res.setTempMemory(1 * (1024 * 1024 * 1024))  # Bytes, the single digit is basically GB)
        self.flat_config = faiss.GpuIndexFlatConfig()
        self.flat_config.device = self.gpu_id


    def build_nn_index(self, database):
        index_cpu = faiss.IndexFlatL2(3)
        index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, index_cpu)
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        D, I = index.search(query, k)

        D_var =torch.from_numpy(np.ascontiguousarray(D))
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))
        if self.gpu_id >= 0:
            D_var = D_var.to(self.device)
            I_var = I_var.to(self.device)

        return D_var, I_var

    def forward(self, predict: T, gt: T) -> TS:
        predict_pc, predict_pcn = predict
        gt_pc, gt_pcn = gt
        if gt_pc.dim() < 3:
            predict_pc, predict_pcn, gt_pc, gt_pcn  = tuple(
                map(lambda x: x.unsqueeze(0), [predict_pc, predict_pcn, gt_pc, gt_pcn]))
        if gt_pc.shape[1] > 3:
            predict_pc, predict_pcn, gt_pc, gt_pcn = tuple(
                map(lambda x: x.permute(0, 2, 1), [predict_pc, predict_pcn, gt_pc, gt_pcn]))
        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()
        predict_pc_np = np.ascontiguousarray(torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy())  # BxMx3
        gt_pc_np = np.ascontiguousarray(torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy())  # BxNx3

        # selected_gt: Bxkx3xM
        selected_gt_by_predict = torch.FloatTensor(predict_pc_size[0], 1, predict_pc_size[1], predict_pc_size[2])
        # selected_predict: Bxkx3xN
        selected_predict_by_gt = torch.FloatTensor(gt_pc_size[0], 1, gt_pc_size[1], gt_pc_size[2])
        # normals
        selected_gt_by_predictn = torch.FloatTensor(predict_pc_size[0], 1, predict_pc_size[1], predict_pc_size[2])
        selected_predict_by_gtn = torch.FloatTensor(gt_pc_size[0], 1, gt_pc_size[1], gt_pc_size[2])

        if self.gpu_id >= 0:
            selected_gt_by_predict = selected_gt_by_predict.to(self.device)
            selected_predict_by_gt = selected_predict_by_gt.to(self.device)
            selected_gt_by_predictn = selected_gt_by_predictn.to(self.device)
            selected_predict_by_gtn = selected_predict_by_gtn.to(self.device)

        # process each batch independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_var = self.search_nn(index_gt, predict_pc_np[i], 1)

            # process nearest k neighbors
            selected_gt_by_predict[i, 0,...] = gt_pc[i].index_select(1, I_var[:, 0])


            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_var = self.search_nn(index_predict, gt_pc_np[i], 1)


            selected_predict_by_gt[i,0,...] = predict_pc[i].index_select(1, I_var[:,0])
            selected_gt_by_predictn[i, 0, ...] = gt_pcn[i].index_select(1, I_var[:, 0])
            selected_predict_by_gtn[i,0,...] = predict_pcn[i].index_select(1, I_var[:,0])

        # compute loss ===================================================
        # selected_gt(Bxkx3xM) vs predict_pc(Bx3xM)
        forward_loss_element = robust_norm(selected_gt_by_predict-predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict))
        loss_source2target_distance = forward_loss_element.mean()

        # normals
        loss_source2target_noramls = - torch.einsum('bkdp,bdp->kp', selected_gt_by_predictn, predict_pcn).mean()

        # selected_predict(Bxkx3xN) vs gt_pc(Bx3xN)
        backward_loss_element = robust_norm(selected_predict_by_gt - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt))  # BxkxN
        loss_target2source_distace = backward_loss_element.mean()

        loss_target2source_noramls = - torch.einsum('bkdp,bdp->kp', selected_predict_by_gtn, gt_pcn).mean()

        return loss_source2target_distance, loss_target2source_distace, loss_source2target_noramls, loss_target2source_noramls

    def __call__(self, predict_pc, gt_pc) -> TS:
        return self.forward(predict_pc, gt_pc)
