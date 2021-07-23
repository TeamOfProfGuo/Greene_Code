
_, _, h, w = x.size()
d0 = self.d_layer0(d)  # [B, 64, h/2, w/2]
x0 = self.layer0(x)  # [B, 64, h/2, w/2]
l0, d0 = self.fuse0(x0, d0)  # [B, 64, h/2, w/2]

d1 = self.d_pool1(d0)  # [B, 64, h/4, w/4]
d1 = self.d_layer1(d1)  # [B, 64, h/4, w/4]
l1 = self.pool1(l0)  # [B, 64, h/4, w/4]
l1 = self.layer1(l1)  # [B, 64, h/4, w/4]
l1, d1= self.fuse1(l1, d1)  # [B, 64, h/4, w/4]

d2 = self.d_layer2(d1)  # [B, 128, h/8, w/8]
l2 = self.layer2(l1)  # [B, 128, h/8, w/8]
l2, d2 = self.fuse2(l2, d2)  # [B, 128, h/8, w/8]

d3 = self.d_layer3(d2)
l3 = self.layer3(l2)  # [B, 256, h/16, w/16]
l3, d3= self.fuse3(l3, d3)  # [B, 256, h/16, w/16]

d4 = self.d_layer4(d3)
l4 = self.layer4(l3)  # [B, 512, h/32, w/32]
l4, _ = self.fuse4(l4, d4)  # [B, 512, h/32, w/32]

y4 = self.up4(l4)  # [B, 256, h/16, w/16]
y3 = self.level_fuse3(y4, l3)

y3 = self.up3(y3)  # [B, 128, h/8, w/8]
y2 = self.level_fuse2(y3, l2)  # [B, 128, h/8, w/8]

y2 = self.up2(y2)  # [B, 64, h/4, w/4]
y1 = self.level_fuse1(y2, l1)  # [B, 64, h/4, w/4]