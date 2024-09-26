import math

def compute_layer_channels_linear(in_channel, trg_channel, num_layers):
    """
    compute the channel list, the channel increase linearly.
    """
    current_channel = in_channel
    channel_list = [current_channel]
    if in_channel < 16:
        in_channel = 0
    step = (trg_channel - in_channel) / num_layers
    for _ in range(num_layers):
        in_channel += step
        channel_list.append(int(in_channel))
    return channel_list

def compute_layer_channels_exp(in_channel, trg_channel, num_layers, divide=1):
    """
    compute the channel list, the channel increase exponentially.
    """
    def _round(x):
        return int(round(x / divide) * divide)

    base = math.exp(math.log(trg_channel / in_channel) / num_layers)
    return [_round(in_channel * base ** i) for i in range(num_layers + 1)]

if __name__ == "__main__":
    res= compute_layer_channels_exp(64, 128, 4, 2)
    print(res)