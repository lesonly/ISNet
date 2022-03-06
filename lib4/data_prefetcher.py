import torch


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_mask, self.next_edge,self.next_region, self.next_region2,_, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_mask = None
            self.next_edge = None
            self.next_region=None
            self.next_region2=None

            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_mask = self.next_mask.cuda(non_blocking=True)
            self.next_edge = self.next_edge.cuda(non_blocking=True)
            self.next_region = self.next_region.cuda(non_blocking=True)
            self.next_region2 = self.next_region2.cuda(non_blocking=True)
            self.next_input = self.next_input.float()  # if need
            self.next_mask = self.next_mask.float()  # if need
            self.next_edge = self.next_edge.float()  # if need
            self.next_region = self.next_region.float()  # if need
            self.next_region2 = self.next_region2.float()  # if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        mask = self.next_mask
        edge = self.next_edge
        region = self.next_region
        region2 = self.next_region2
        self.preload()
        return input, mask, edge,region,region2
