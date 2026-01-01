
# 5. 推論と結果の可視化
model.eval()
test_images = next(iter(train_loader)).to(device)
with torch.no_grad():
    L, S = model(test_images)

# 最初の1枚を表示
idx = 0
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(test_images[idx].cpu().squeeze(), cmap='gray')
plt.title("Original (Metal Surface)")
plt.subplot(1, 3, 2)
plt.imshow(L[idx].cpu().squeeze(), cmap='gray')
plt.title("Low-Rank (Background)")
plt.subplot(1, 3, 3)
plt.imshow(torch.abs(S[idx]).cpu().squeeze(), cmap='hot')
plt.title("Sparse (Detected Defect)")
plt.show()