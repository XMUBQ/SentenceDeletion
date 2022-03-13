import argparse

parser = argparse.ArgumentParser(description='No description')
parser.add_argument('--basemodel', type=str, default='joint_learning', help="base, concat, joint learning")
parser.add_argument('--gamma', type=float, help='gamma', default=1.5)
parser.add_argument('--testmode', type=int, help='normal training:0; test:1; valid:2', default=0)
parser.add_argument('--testepoch', type=int, help='epoch to evaluate', default=0)
parser.add_argument('--loadepoch', type=int, help='load before training', default=-1)
parser.add_argument('--T', type=int, help='total training epochs', default=10)
parser.add_argument('--cuda', type=int, default=0)

args = parser.parse_args()
print(args)
