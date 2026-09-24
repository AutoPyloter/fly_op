import torch
import numpy as np
import pandas as pd
import time
from scipy import sparse
import os

# FlyOpt Kütüphaneleri
from flyopt.variants.rate_brain import RateBrain, RateBrainConfig, build_subgraph_bfs, select_connected_encode_decode

# ==========================================
# 1. BENCHMARK (TEST) FONKSİYONLARI
# ==========================================
# Sphere: Basit, tek bir çukuru olan kolay optimizasyon problemi
def sphere(x):
    return torch.sum(x**2, dim=1)

# Rastrigin: Çok fazla sahte çukuru (tuzakları) olan, zorlu biyolojik hayatta kalma simülasyonu
def rastrigin(x):
    A = 10
    return A * x.shape[1] + torch.sum(x**2 - A * torch.cos(2 * np.pi * x), dim=1)


# ==========================================
# 2. PİYASA STANDARDI: PSO (Parçacık Sürüsü Optimizasyonu)
# ==========================================
def pso_optimize(func, dim=2, num_particles=1000, iters=100, device='cpu'):
    print(f"Piyasa Algoritması (PSO) çalışıyor... ({iters} İterasyon)")
    x = torch.rand((num_particles, dim), device=device) * 10 - 5
    v = torch.zeros_like(x)
    pbest = x.clone()
    pbest_obj = func(x)
    gbest = pbest[torch.argmin(pbest_obj)].clone()
    gbest_obj = torch.min(pbest_obj)

    w, c1, c2 = 0.5, 1.5, 1.5
    history = []

    start_time = time.time()
    for i in range(iters):
        r1 = torch.rand((num_particles, dim), device=device)
        r2 = torch.rand((num_particles, dim), device=device)
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = x + v
        
        obj = func(x)
        better_mask = obj < pbest_obj
        pbest[better_mask] = x[better_mask]
        pbest_obj[better_mask] = obj[better_mask]
        
        if torch.min(pbest_obj) < gbest_obj:
            gbest = pbest[torch.argmin(pbest_obj)].clone()
            gbest_obj = torch.min(pbest_obj)
            
        history.append(gbest_obj.item())
        
    calc_time = time.time() - start_time
    return history, calc_time


# ==========================================
# 3. BİYOLOJİK MOTOR (FlyOpt T=20 Erkek Sinek)
# ==========================================
def fly_optimize(func, brain, dim=2, num_particles=1000, iters=100, device='cpu'):
    print(f"Biyolojik Sinek Algoritması (FlyOpt T=20) çalışıyor... ({iters} İterasyon)")
    # Sineğin başlangıç konumları
    x = torch.rand((num_particles, dim), device=device) * 10 - 5
    best_obj = func(x)
    best_x = x.clone()
    
    # Sinek beyin girdisi için padding (eğer 4 input istiyorsa)
    input_pad = torch.zeros((num_particles, 4 - dim), device=device)
    
    history = []
    start_time = time.time()
    
    # Sinek T=20 adımlık iç güdüsüyle çevreyi tarıyor ve adım (delta) üretiyor
    with torch.no_grad():
        for i in range(iters):
            # Sineğin çevresel algısı (Mevcut konumu beyne gönderiyoruz)
            env_input = torch.cat([x, input_pad], dim=1)
            
            # Beyin T=20 adım düşünür ve bir Rota (Step Vector) çizer
            fly_step = brain(env_input)
            
            # Sadece ilk iki boyutu hareket yönü olarak kullan
            move_vector = fly_step[:, :dim] * 0.1 # 0.1 adımlama hızı
            
            # Yeni konuma uç
            x_new = x - move_vector 
            
            # Yeni konum eskisinden daha iyi mi (Hayatta kalma içgüdüsü)?
            new_obj = func(x_new)
            better_mask = new_obj < best_obj
            
            # Eğer daha iyiyse oraya yerleş (Foraging başarısı)
            x[better_mask] = x_new[better_mask]
            best_obj[better_mask] = new_obj[better_mask]
            
            # Sürünün en iyi bireyini kaydet
            gbest_obj = torch.min(best_obj)
            history.append(gbest_obj.item())
            
    calc_time = time.time() - start_time
    return history, calc_time


# ==========================================
# ANA ÇALIŞTIRICI VE RAPORLAMA
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- FLYOPT BENCHMARK KARŞILAŞTIRMASI BAŞLIYOR (Donanım: {device}) ---")

    # 1. Erkek Sinek (Male CNS) Beynini Yükle
    project_path = "C:/projeler/fly_op"
    DATA_PROCESSED = f"{project_path}/data/processed"
    print("Erkek sinek connectome (ağ) yükleniyor...")
    base_weights = sparse.load_npz(f"{DATA_PROCESSED}/malecns_adjacency.npz")
    afferent = np.load(f"{DATA_PROCESSED}/malecns_afferent_indices.npy")
    efferent = np.load(f"{DATA_PROCESSED}/malecns_efferent_indices.npy")

    encode_full, decode_full = select_connected_encode_decode(
        base_weights, afferent, efferent, n_encode=4, n_decode=4, max_hops=6, seed=9000
    )
    sub_real, encode_idx, decode_idx, _ = build_subgraph_bfs(base_weights, encode_full, decode_full, 3000, seed=9000)

    # T=20 ile ağır biyolojik motor
    cfg = RateBrainConfig(dim=2, n_readout=len(decode_idx), T=20, decode_scale=0.5, train_gain=True)
    brain = RateBrain(sub_real, encode_idx, decode_idx, cfg, seed=42).to(device)
    
    # 2. Testleri Koştur (1000 Ajan, 150 İterasyon)
    NUM_AGENTS = 1000
    ITERS = 150
    
    # RASTRİGİN TESTİ (ZORLU GÖREV)
    print("\n[GÖREV 1] Rastrigin Fonksiyonu (Çok tuzaklı yüzey)")
    pso_hist_r, pso_time_r = pso_optimize(rastrigin, dim=2, num_particles=NUM_AGENTS, iters=ITERS, device=device)
    fly_hist_r, fly_time_r = fly_optimize(rastrigin, brain, dim=2, num_particles=NUM_AGENTS, iters=ITERS, device=device)
    
    # SPHERE TESTİ (KOLAY GÖREV)
    print("\n[GÖREV 2] Sphere Fonksiyonu (Pürüzsüz yüzey)")
    pso_hist_s, pso_time_s = pso_optimize(sphere, dim=2, num_particles=NUM_AGENTS, iters=ITERS, device=device)
    fly_hist_s, fly_time_s = fly_optimize(sphere, brain, dim=2, num_particles=NUM_AGENTS, iters=ITERS, device=device)

    # 3. Sonuçları Tabloya (Rapor) Dök
    df = pd.DataFrame({
        "İterasyon": list(range(1, ITERS+1)),
        "PSO_Rastrigin_Skor": pso_hist_r,
        "FlyOpt_Rastrigin_Skor": fly_hist_r,
        "PSO_Sphere_Skor": pso_hist_s,
        "FlyOpt_Sphere_Skor": fly_hist_s
    })
    
    report_path = f"{project_path}/benchmark_karsilastirma_raporu.csv"
    df.to_csv(report_path, index=False)
    
    print("\n" + "="*50)
    print("🏆 BÜYÜK KARŞILAŞTIRMA SONUÇLARI 🏆")
    print("="*50)
    print(f"PSO (Rastrigin) En İyi Skor : {pso_hist_r[-1]:.4f} (Süre: {pso_time_r:.2f} sn)")
    print(f"FlyOpt (Rastrigin) En İyi Skor: {fly_hist_r[-1]:.4f} (Süre: {fly_time_r:.2f} sn)")
    print("-" * 50)
    print(f"PSO (Sphere) En İyi Skor    : {pso_hist_s[-1]:.4f} (Süre: {pso_time_s:.2f} sn)")
    print(f"FlyOpt (Sphere) En İyi Skor   : {fly_hist_s[-1]:.4f} (Süre: {fly_time_s:.2f} sn)")
    print("="*50)
    print(f"✅ Bütün iterasyon adımları, skorlar ve karşılaştırma tablosu kaydedildi: {report_path}")
    print("Bu CSV dosyasını Colab'da 'sns.lineplot' ile çizdirip iki algoritmanın düşüş (öğrenme) hızını yarıştırabilirsiniz!")

if __name__ == "__main__":
    main()
