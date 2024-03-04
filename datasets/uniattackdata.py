from .text_folder_dataset import TextFolderDataset
import os

class UniAttackData(TextFolderDataset):
    def __init__(self, root_folder, protocol=None, subset=None, phase="phase1", transform=None, labels=True):
        assert protocol in ["p1", "p2.1", "p2.2"]
        assert subset in ["train", "dev", "test"]
        assert phase in ["phase1", "phase2"]
        if (subset == "test") and not (phase == "phase2"):
            raise Exception("Test set only available in phase2")

        txt_file_pth = os.path.join(phase,protocol,f"{subset}{'_label' if labels else ''}.txt")
        super().__init__(root_folder, phase, txt_file_pth, transform=transform)
