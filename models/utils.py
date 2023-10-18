"""
@Date  : 2022/12/18
@Time  : 15:18
@Author: Ziyang Huang
@Email : huangzy0312@gmail.com
"""
import torch


def decimal_to_bits(x, bits):
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    device = x.device

    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device)
    x = x.unsqueeze(dim=-1)
    # x = rearrange(x, 'b l -> b l 1')

    bits = ((x & mask) != 0).float()
    bits = bits * 2 - 1
    return bits


def bits_to_decimal(x, bits):
    """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
    device = x.device

    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int32)

    # dec = reduce(x * mask, 'b l d -> b l', 'sum')
    dec = (x * mask).sum(dim=-1)
    # [bsz, len]
    return dec


if __name__ == '__main__':
    x = torch.randint(0, 8, [8, 12, 12])
    bits = decimal_to_bits(x, 3)
    decimal = bits_to_decimal(bits * 3, 3)
    print(decimal)
