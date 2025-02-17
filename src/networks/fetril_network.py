import argparse
import torch.nn
from sklearn.svm import LinearSVC

from networks.cil_network import CIL_Net


class LinearSVCHead(torch.nn.Module):
    def __init__(self, svc: LinearSVC, num_classes):
        super().__init__()
        self.svc = svc
        self.num_classes = num_classes

    def forward(self, x):
        svc_input = x.cpu().detach().numpy()
        svc_result = self.svc.predict(svc_input)

        # Generate pseudo confidences:
        confidences = torch.zeros((x.size(0), self.num_classes), device=x.device, dtype=torch.float)
        for i, res in enumerate(svc_result):
            confidences[i, res] = 1.0
        return confidences

class FetrilNet(CIL_Net):
    def __init__(self, network_name, pretrained, remove_existing_head, not_use_l2_norm, classifier_type,
                 svc_tolerance, svc_regularization):
        super().__init__(network_name, pretrained, remove_existing_head)
        self.l2_norm = not not_use_l2_norm
        self.classifier_type = classifier_type
        self._svc_tol = svc_tolerance
        self._svc_reg = svc_regularization

    @staticmethod
    def extra_parser(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--not-use-l2-norm", action="store_true")
        parser.add_argument("--classifier-type", default="fc", choices=["fc", "linear_svc"])
        parser.add_argument("--svc-tolerance", type=float, default=0.0001)
        parser.add_argument("--svc-regularization", type=float, default=1.0)
        return parser.parse_known_args(args)

    def forward(self, x, return_features=False):
        features = self.model(x)
        if self.l2_norm:
            norm = torch.nn.functional.normalize(features, p=2, dim=1)
        else:
            norm = features
        y = self.head(norm)

        if return_features:
            return y, features
        return y

    def modify_head(self, num_outputs):
        if type(self.head) == LinearSVCHead:
            assert self.head.num_classes <= num_outputs
            self.head.num_classes = num_outputs
        else:
            super().modify_head(num_outputs)

    def replace_svc_head(self, features: torch.Tensor, targets: torch.Tensor):
        num_outputs = self.head.num_classes if type(self.head) == LinearSVCHead else self.head.out_features
        # Train the linear svc and use it as a head
        svc = LinearSVC(penalty='l2', dual=False, tol=self._svc_tol, C=self._svc_reg, multi_class='ovr',
                        fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0)
        svc.fit(features.numpy(), targets.numpy())
        self.head = LinearSVCHead(svc, num_classes=num_outputs)  # Replace with most recent Linear SVC Head
