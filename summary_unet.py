## -- This is to test the u-net model with synthetic dataset
def test():
    import torch
    from unet import UNet
    model = UNet(in_channels=1,
                out_channels=1,
                n_blocks=5,
                start_filters=32,
                activation='relu',
                normalization='batch',
                conv_mode='same',
                dim=3)

    # Create a random dataset 
    x = torch.randn(size=(1, 1, 96, 128, 96), dtype=torch.float32)

    with torch.no_grad():
        out = model(x)

    print(f'Out: {out.shape}')
    print(f'In: {x.shape}')


    # Print summary of the model 
    from torchinfo import summary
    summary(model, input_size=(1, 1, 96, 128, 96))

if __name__ == "__main__":
    test()

