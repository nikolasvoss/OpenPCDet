from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # only return features_dict if vales are present, otherwise compatibility issues occur
            required_keys = ["time_wo_ent", "time_entropy", "memory_con_wo_ent", "memory_con",
                             "multi_scale_3d_features"]

            if any(key in batch_dict for key in required_keys):
                features_dict = {key: batch_dict[key] for key in required_keys if key in batch_dict}
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts, features_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
            # features_dict = {"time_wo_ent": batch_dict['time_wo_ent'],
            #                  "time_entropy": batch_dict['time_entropy'],
            #                  "memory_con_wo_ent": batch_dict['memory_con_wo_ent'],
            #                  "memory_con": batch_dict['memory_con'],
            #                  "multi_scale_3d_features": batch_dict['multi_scale_3d_features']}
            # pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # return pred_dicts, recall_dicts, features_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
