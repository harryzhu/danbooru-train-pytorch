# Update:
the last layer's output_channels has been modified from 9176 to 6684 because I have removed some NSFW lables.

## Current: out_channels=6684
```
self.n_Conv_178 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=6684, bias=False)
```

## Original: out_channels=9176
```
self.n_Conv_178 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=9176, bias=False)
```
