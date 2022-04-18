from cgi import print_form
import torch

controls_masks = []
controls = torch.tensor([2, 2])
controls_b1 = (controls == 2)
print(controls_b1)
controls_b1 = torch.tensor(controls_b1, dtype=torch.float32)
print(controls_b1)

controls_b1 = torch.cat([controls_b1] * 3, 1)
print(controls_b1)
    
controls_masks.append(controls_b1)

print(controls_masks)