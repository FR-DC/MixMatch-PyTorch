import torch


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


#%%
bn = torch.nn.BatchNorm1d(1, momentum=0.5)
bn(torch.ones(2, 1, 1))
bn(torch.ones(2, 1, 1))
bn(torch.zeros(2, 1, 1))
print(bn.running_mean)
bn = torch.nn.BatchNorm1d(1, momentum=0.5)
bn(torch.zeros(2, 1, 1))
bn(torch.ones(2, 1, 1))
bn(torch.ones(2, 1, 1))
print(bn.running_mean)
# bn(torch.ones(2, 3, 32, 32))
#%%

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_clipboard()
#%%

plt.figure(figsize=(6, 3))
df_ = df.set_index("Epoch")
df_ = df_.rename(columns={"No Interleave Test": "No Interleave, Across Augs",
                          "Batch Test": "Batch, Across Augs, LR * 3"})
df_[['Original Test', 'No Interleave, Across Augs', 'Batch, Across Augs, LR * 3']].plot(ax=plt.gca())


plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CIFAR-10 Performance with different interleave methods")
plt.tight_layout()
plt.show()


