from networks.pass_network import PASS_Net


class PRAKA_Net(PASS_Net):
    def __init__(self, network_name, pretrained, remove_existing_head):
        super().__init__(network_name, pretrained, remove_existing_head)

    def collapse_orientation_head(self):
        raise RuntimeError("The collapse of the orientation head is not supported in the PRAKA method rather")
