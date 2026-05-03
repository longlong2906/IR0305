import os
import time
import numpy as np
import faiss

def evaluate_recall(ground_truth_indices, test_indices):
    """
    Tính trung bình recall của batch test so với ground truth.
    ground_truth_indices: mảng shape (num_queries, k)
    test_indices: mảng shape (num_queries, k)
    """
    recalls = []
    for gt, test in zip(ground_truth_indices, test_indices):
        intersection = np.intersect1d(gt, test)
        recall = len(intersection) / len(gt)
        recalls.append(recall)
    return np.mean(recalls)

def main():
    if not os.path.exists("embeddings_50k.npy"):
        print("Không tìm thấy file embeddings_50k.npy. Vui lòng chạy lại index_builder.py trước.")
        return
        
    print("Loading embeddings_50k.npy...")
    xb = np.load("embeddings_50k.npy").astype('float32')
    dataset_size, d = xb.shape
    
    # Sinh 1000 queries ngẫu nhiên
    np.random.seed(42)
    num_queries = 1000
    query_idxs = np.random.choice(dataset_size, num_queries, replace=False)
    xq = xb[query_idxs]
    
    k_vals = [10, 50, 100]
    max_k = 100
    
    # Tính Ground Truth bằng Flat
    print("Building Flat Index for Ground Truth...")
    index_flat = faiss.IndexFlatIP(d)
    index_flat.add(xb)
    _, I_flat = index_flat.search(xq, max_k)
    
    # ----------------------------------------------------
    # TUNING HNSW (efSearch)
    # ----------------------------------------------------
    print("\n--- Tuning HNSW (efSearch) ---")
    index_hnsw = faiss.IndexHNSWFlat(d, 32)
    index_hnsw.add(xb)
    
    hnsw_results = []
    efSearch_values = [16, 32, 64, 128, 256]
    for efSearch in efSearch_values:
        index_hnsw.hnsw.efSearch = efSearch
        
        start_t = time.time()
        _, I_hnsw = index_hnsw.search(xq, max_k)
        search_t = (time.time() - start_t) * 1000 # ms
        
        recall_10 = evaluate_recall(I_flat[:, :10], I_hnsw[:, :10])
        recall_50 = evaluate_recall(I_flat[:, :50], I_hnsw[:, :50])
        recall_100 = evaluate_recall(I_flat[:, :100], I_hnsw[:, :100])
        
        hnsw_results.append((efSearch, search_t, recall_10, recall_50, recall_100))
        print(f"efSearch={efSearch:3d} | Time: {search_t:6.1f}ms | R@10: {recall_10:.4f} | R@50: {recall_50:.4f} | R@100: {recall_100:.4f}")

    # ----------------------------------------------------
    # TUNING IVF (nprobe)
    # ----------------------------------------------------
    print("\n--- Tuning IVF (nprobe) ---")
    nlist = 400
    quantizer_ivf = faiss.IndexFlatIP(d)
    index_ivf = faiss.IndexIVFFlat(quantizer_ivf, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(xb)
    index_ivf.add(xb)
    
    ivf_results = []
    nprobe_values = [1, 5, 10, 20, 50, 100]
    for nprobe in nprobe_values:
        index_ivf.nprobe = nprobe
        
        start_t = time.time()
        _, I_ivf = index_ivf.search(xq, max_k)
        search_t = (time.time() - start_t) * 1000
        
        recall_10 = evaluate_recall(I_flat[:, :10], I_ivf[:, :10])
        recall_50 = evaluate_recall(I_flat[:, :50], I_ivf[:, :50])
        recall_100 = evaluate_recall(I_flat[:, :100], I_ivf[:, :100])
        
        ivf_results.append((nprobe, search_t, recall_10, recall_50, recall_100))
        print(f"nprobe={nprobe:3d} | Time: {search_t:6.1f}ms | R@10: {recall_10:.4f} | R@50: {recall_50:.4f} | R@100: {recall_100:.4f}")

    # ----------------------------------------------------
    # TUNING IVF+PQ (nprobe)
    # ----------------------------------------------------
    print("\n--- Tuning IVF+PQ (nprobe) ---")
    m = 8
    quantizer_pq = faiss.IndexFlatIP(d)
    index_pq = faiss.IndexIVFPQ(quantizer_pq, d, nlist, m, 8)
    index_pq.train(xb)
    index_pq.add(xb)
    
    pq_results = []
    for nprobe in nprobe_values:
        index_pq.nprobe = nprobe
        
        start_t = time.time()
        _, I_pq = index_pq.search(xq, max_k)
        search_t = (time.time() - start_t) * 1000
        
        recall_10 = evaluate_recall(I_flat[:, :10], I_pq[:, :10])
        recall_50 = evaluate_recall(I_flat[:, :50], I_pq[:, :50])
        recall_100 = evaluate_recall(I_flat[:, :100], I_pq[:, :100])
        
        pq_results.append((nprobe, search_t, recall_10, recall_50, recall_100))
        print(f"nprobe={nprobe:3d} | Time: {search_t:6.1f}ms | R@10: {recall_10:.4f} | R@50: {recall_50:.4f} | R@100: {recall_100:.4f}")

    # XUẤT RA MARKDOWN
    generate_tuning_report(hnsw_results, ivf_results, pq_results)

def generate_tuning_report(hnsw, ivf, pq, filepath="tuning_report.md"):
    md = "# Báo cáo Tuning tham số FAISS\n\n"
    md += "*Dataset: 50,000 vectors | Số lượng Queries: 1,000*\n\n"
    
    # Bảng HNSW
    md += "## 1. Tuning HNSW (`efSearch`)\n\n"
    md += "| efSearch | Search Time 1000q (ms) | Recall@10 | Recall@50 | Recall@100 |\n"
    md += "|---|---|---|---|---|\n"
    for r in hnsw:
        md += f"| **{r[0]}** | {r[1]:.2f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |\n"
    
    # Bảng IVF
    md += "\n## 2. Tuning IVF (`nprobe`)\n\n"
    md += "*nlist = 400*\n\n"
    md += "| nprobe | Search Time 1000q (ms) | Recall@10 | Recall@50 | Recall@100 |\n"
    md += "|---|---|---|---|---|\n"
    for r in ivf:
        md += f"| **{r[0]}** | {r[1]:.2f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |\n"
        
    # Bảng IVF+PQ
    md += "\n## 3. Tuning IVF+PQ (`nprobe`)\n\n"
    md += "*nlist = 400, m = 8*\n\n"
    md += "| nprobe | Search Time 1000q (ms) | Recall@10 | Recall@50 | Recall@100 |\n"
    md += "|---|---|---|---|---|\n"
    for r in pq:
        md += f"| **{r[0]}** | {r[1]:.2f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |\n"
        
    # Phân tích
    md += """
## Diễn giải sự đánh đổi (Trade-off) giữa độ chính xác và tốc độ

### `efSearch` (HNSW)
- **Cơ chế:** Tham số `efSearch` quyết định thuật toán sẽ giữ bao nhiêu ứng viên trong hàng đợi (queue) ở quá trình duyệt đồ thị để tìm đường đến điểm gần nhất.
- **Trade-off:** Việc tăng `efSearch` sẽ giúp Recall tăng lên tiệm cận 1.0 (vì khả năng bỏ sót đường đi ngắn nhất giảm đi), nhưng bù lại làm cho **Search Time tăng cao hơn** do phải tính toán và duy trì nhiều nút hơn trong hàng đợi.

### `nprobe` (IVF và IVF+PQ)
- **Cơ chế:** Thuật toán IVF (K-Means) đã chia 50,000 vectors vào `nlist = 400` cụm. Khi có query tới, nó sẽ xác định cụm nào có trung tâm (centroid) gần với query nhất và quét bên trong cụm đó. Tham số `nprobe` quyết định hệ thống sẽ quét bao nhiêu cụm gần nhất.
- **Trade-off:** 
  - Khi `nprobe = 1`, tốc độ cực kì nhanh (vì chỉ quét 1/400 tổng số vectors), nhưng Recall rất thấp vì có khả năng điểm gần nhất thực sự lại nằm ở rìa cụm lân cận.
  - Khi `nprobe` tăng dần (ví dụ lên 50 hay 100), Recall tăng mạnh, nhưng **Search Time cũng tăng** vì số vector phải duyệt tăng tỷ lệ thuận với số cụm. (Nếu nprobe = 400, nó trở về tương đương với vét cạn Flat).
- **IVF vs IVF+PQ:** Ở bảng IVF, khi nprobe đủ lớn, Recall có thể đạt tới 1.0. Nhưng với **IVF+PQ**, vì các vector bên trong cụm đã bị nén (quantization) làm giảm độ chính xác, dù bạn có tăng nprobe lên tối đa thì Recall cũng sẽ chững lại và **không thể đạt 1.0**. Sự hi sinh độ chính xác này là bắt buộc để đổi lấy lợi thế cực lớn về Memory (Size).
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nBáo cáo chi tiết đã được xuất ra {filepath}")

if __name__ == "__main__":
    main()
