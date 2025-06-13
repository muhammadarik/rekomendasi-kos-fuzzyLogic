#REKOMENDASI TEMPAT KOS DENGAN FUZZY LOGIC DAN DETEKSI OUTLIER
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class KosRecommendationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Rekomendasi Tempat Kos")
        self.root.geometry("900x700")
        
        # Data contoh tempat kos
        self.kos_data = self.generate_sample_data(50)
        
        # Variabel fuzzy
        self.jarak = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'jarak')
        self.harga = ctrl.Antecedent(np.arange(350000, 3000001, 50000), 'harga')
        self.fasilitas = ctrl.Antecedent(np.arange(0, 11, 1), 'fasilitas')
        self.rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')

        show_graph_btn = ttk.Button(self.root, text="Tampilkan Grafik Data", command=self.show_data_graph)
        show_graph_btn.pack(pady=5)

        # Variabel untuk menyimpan referensi canvas grafik
        self.graph_canvas = None
        
        # Fungsi keanggotaan
        self.setup_fuzzy_membership()
        
        # Rules fuzzy
        self.setup_fuzzy_rules()
        
        # Sistem kontrol fuzzy
        self.rekomendasi_ctrl = ctrl.ControlSystem(self.rules)
        self.rekomendasi_system = ctrl.ControlSystemSimulation(self.rekomendasi_ctrl)
        
        # Deteksi outlier
        self.detect_outliers()
        
        # GUI Components
        self.setup_gui()
        
    def generate_sample_data(self, n):
        """Generate sample kos data with some outliers"""
        data = []
        for _ in range(n):
            # Normal data
            jarak = round(random.uniform(0.1, 10), 1)
            harga = random.choice([
                random.randint(350000, 450000), 
                random.randint(450000, 600000),
                random.randint(600000, 1000000)
            ])
            fasilitas = random.randint(3, 9)
            
            # Add some outliers (5% chance)
            if random.random() < 0.05:
                if random.random() < 0.5:
                    jarak = round(random.uniform(15, 20), 1)  # Very far
                    harga = random.randint(3000000, 5000000)  # Very expensive
                else:
                    fasilitas = random.randint(0, 2)  # Very poor facilities
            
            data.append({
                'nama': f"Kos {random.choice(['A', 'B', 'C'])}-{random.randint(1, 100)}",
                'jarak': jarak,
                'harga': harga,
                'fasilitas': fasilitas,
                'alamat': f"Jalan {random.choice(['Mawar', 'Melati', 'Anggrek'])} No. {random.randint(1, 50)}"
            })
        return pd.DataFrame(data)

    def show_data_graph(self):
        """Menampilkan grafik distribusi data kos"""
        # Hapus canvas grafik sebelumnya jika ada
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
        
        # Buat window baru untuk grafik
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Visualisasi Data Kos")
        graph_window.geometry("900x600")
        
        # Buat figure matplotlib
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        
        # Grafik 1: Distribusi Jarak
        jarak_data = self.kos_data['jarak']
        ax1.hist(jarak_data, bins=15, color='skyblue', edgecolor='black')
        ax1.set_title('Distribusi Jarak Kos dari Kampus (km)')
        ax1.set_xlabel('Jarak (km)')
        ax1.set_ylabel('Jumlah Kos')
        
        # Grafik 2: Distribusi Harga
        harga_data = self.kos_data['harga'] / 1000000  # Konversi ke juta
        ax2.hist(harga_data, bins=15, color='lightgreen', edgecolor='black')
        ax2.set_title('Distribusi Harga Sewa Kos (juta Rp)')
        ax2.set_xlabel('Harga (juta Rp)')
        ax2.set_ylabel('Jumlah Kos')
        
        # Grafik 3: Distribusi Fasilitas
        fasilitas_data = self.kos_data['fasilitas']
        ax3.hist(fasilitas_data, bins=10, color='salmon', edgecolor='black')
        ax3.set_title('Distribusi Tingkat Fasilitas Kos (1-10)')
        ax3.set_xlabel('Tingkat Fasilitas')
        ax3.set_ylabel('Jumlah Kos')
        
        # Atur layout
        fig.tight_layout()
        
        # Embed grafik di Tkinter
        self.graph_canvas = FigureCanvasTkAgg(fig, master=graph_window)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tambahkan tombol untuk menampilkan grafik rekomendasi
        rec_graph_btn = ttk.Button(graph_window, text="Tampilkan Grafik Rekomendasi", 
                                 command=lambda: self.show_recommendation_graph(graph_window))
        rec_graph_btn.pack(pady=5)
    
    def show_recommendation_graph(self, parent_window):
        """Menampilkan grafik skor rekomendasi"""
        # Hitung skor rekomendasi untuk semua data
        recommendations = []
        for _, row in self.kos_data.iterrows():
            score, _ = self.calculate_recommendation(row['jarak'], row['harga'], row['fasilitas'])
            recommendations.append(score)
        
        # Buat window baru
        rec_window = tk.Toplevel(parent_window)
        rec_window.title("Visualisasi Skor Rekomendasi")
        rec_window.geometry("700x500")
        
        # Buat figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        # Grafik 1: Histogram Skor Rekomendasi
        ax1.hist(recommendations, bins=15, color='purple', edgecolor='black')
        ax1.set_title('Distribusi Skor Rekomendasi')
        ax1.set_xlabel('Skor Rekomendasi (0-100)')
        ax1.set_ylabel('Jumlah Kos')
        
        # Grafik 2: Scatter Plot Harga vs Rekomendasi
        harga_juta = self.kos_data['harga'] / 1000000
        ax2.scatter(harga_juta, recommendations, c='orange', alpha=0.6)
        ax2.set_title('Hubungan Harga dan Skor Rekomendasi')
        ax2.set_xlabel('Harga (juta Rp)')
        ax2.set_ylabel('Skor Rekomendasi')
        ax2.grid(True)
        
        fig.tight_layout()
        
        # Embed grafik di Tkinter
        canvas = FigureCanvasTkAgg(fig, master=rec_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_fuzzy_membership(self):
        """Setup fuzzy membership functions"""
        # Jarak membership functions
        self.jarak['dekat'] = fuzz.trimf(self.jarak.universe, [0.1, 0.1, 4])
        self.jarak['sedang'] = fuzz.trimf(self.jarak.universe, [2, 5, 6])
        self.jarak['jauh'] = fuzz.trimf(self.jarak.universe, [4, 10, 10])
        
        # Harga membership functions
        self.harga['murah'] = fuzz.trimf(self.harga.universe, [350000, 350000, 700000])
        self.harga['sedang'] = fuzz.trimf(self.harga.universe, [500000, 1000000, 1500000])
        self.harga['mahal'] = fuzz.trimf(self.harga.universe, [1200000, 3000000, 3000000])
        
        # Fasilitas membership functions
        self.fasilitas['minimal'] = fuzz.trimf(self.fasilitas.universe, [0, 2, 5])
        self.fasilitas['cukup'] = fuzz.trimf(self.fasilitas.universe, [3, 5, 8])
        self.fasilitas['lengkap'] = fuzz.trimf(self.fasilitas.universe, [6, 10, 10])
        
        # Rekomendasi membership functions
        self.rekomendasi['buruk'] = fuzz.trimf(self.rekomendasi.universe, [0, 0, 40])
        self.rekomendasi['cukup'] = fuzz.trimf(self.rekomendasi.universe, [30, 50, 70])
        self.rekomendasi['baik'] = fuzz.trimf(self.rekomendasi.universe, [60, 100, 100])
    
    def setup_fuzzy_rules(self):
        """Setup fuzzy rules"""
        self.rules = [
            ctrl.Rule(self.jarak['dekat'] & self.harga['murah'] & self.fasilitas['minimal'], self.rekomendasi['cukup']),
            ctrl.Rule(self.jarak['dekat'] & self.harga['sedang'] & self.fasilitas['minimal'], self.rekomendasi['cukup']),
            ctrl.Rule(self.jarak['sedang'] & self.harga['murah'] & self.fasilitas['minimal'], self.rekomendasi['cukup']),
        
            # Rules untuk fasilitas cukup
            ctrl.Rule(self.jarak['dekat'] & self.harga['murah'] & self.fasilitas['cukup'], self.rekomendasi['baik']),
        
            # Rules untuk fasilitas lengkap
            ctrl.Rule(self.jarak['dekat'] & self.harga['murah'] & self.fasilitas['lengkap'], self.rekomendasi['baik']),
            # Tambahkan lebih banyak rule untuk mencakup lebih banyak kombinasi
            ctrl.Rule(self.jarak['dekat'] & self.harga['murah'] & self.fasilitas['minimal'], self.rekomendasi['cukup']),
            ctrl.Rule(self.jarak['dekat'] & self.harga['murah'] & self.fasilitas['cukup'], self.rekomendasi['baik']),
            # Rules for good recommendation
            ctrl.Rule(self.jarak['dekat'] & self.harga['murah'] & self.fasilitas['lengkap'], self.rekomendasi['baik']),
            ctrl.Rule(self.jarak['dekat'] & self.harga['sedang'] & self.fasilitas['lengkap'], self.rekomendasi['baik']),
            ctrl.Rule(self.jarak['sedang'] & self.harga['murah'] & self.fasilitas['lengkap'], self.rekomendasi['baik']),
            
            # Rules for moderate recommendation
            ctrl.Rule(self.jarak['dekat'] & self.harga['mahal'] & self.fasilitas['lengkap'], self.rekomendasi['cukup']),
            ctrl.Rule(self.jarak['sedang'] & self.harga['sedang'] & self.fasilitas['cukup'], self.rekomendasi['cukup']),
            ctrl.Rule(self.jarak['jauh'] & self.harga['murah'] & self.fasilitas['lengkap'], self.rekomendasi['cukup']),
            
            # Rules for bad recommendation
            ctrl.Rule(self.jarak['jauh'] & self.harga['mahal'] & self.fasilitas['minimal'], self.rekomendasi['buruk']),
            ctrl.Rule(self.jarak['sedang'] & self.harga['mahal'] & self.fasilitas['minimal'], self.rekomendasi['buruk']),
            ctrl.Rule(self.jarak['jauh'] & self.harga['sedang'] & self.fasilitas['minimal'], self.rekomendasi['buruk'])
        ]
    
    def detect_outliers(self):
        """Detect and mark outliers in the data"""
        # Simple outlier detection based on our criteria
        self.kos_data['outlier'] = False
        
        for idx, row in self.kos_data.iterrows():
            if (row['jarak'] > 10 or 
                row['harga'] > 3000000 or 
                row['fasilitas'] < 3):
                self.kos_data.at[idx, 'outlier'] = True
    
    def calculate_recommendation(self, jarak, harga, fasilitas):
        """Calculate recommendation score with improved error handling"""
        try:
            # Normalisasi input
            fasilitas = max(0, min(10, fasilitas))
        
            self.rekomendasi_system.input['jarak'] = jarak
            self.rekomendasi_system.input['harga'] = harga
            self.rekomendasi_system.input['fasilitas'] = fasilitas
        
            try:
                self.rekomendasi_system.compute()
                score = self.rekomendasi_system.output['rekomendasi']
            except:
                # Jika tidak ada rule yang teraktivasi, beri nilai default
                score = 30  # Nilai default untuk fasilitas minimal
        
            # Classify based on score
            if score >= 70:
                return score, "Sangat Direkomendasikan"
            elif score >= 40:
                return score, "Cukup Direkomendasikan"
            else:
                return score, "Tidak Direkomendasikan"
        except Exception as e:
            print(f"Error: {str(e)}")
            return 30, "Rekomendasi Dasar (fasilitas minimal)"

    
    def setup_gui(self):
        """Setup the GUI components"""
        # Frame for input
        input_frame = ttk.LabelFrame(self.root, text="Input Kriteria", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Jarak input
        ttk.Label(input_frame, text="Jarak dari kampus (km):").grid(row=0, column=0, sticky=tk.W)
        self.jarak_entry = ttk.Entry(input_frame)
        self.jarak_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Harga input
        ttk.Label(input_frame, text="Harga sewa (Rp):").grid(row=1, column=0, sticky=tk.W)
        self.harga_entry = ttk.Entry(input_frame)
        self.harga_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Fasilitas input
        ttk.Label(input_frame, text="Tingkat fasilitas (1-10):").grid(row=2, column=0, sticky=tk.W)
        self.fasilitas_entry = ttk.Entry(input_frame)
        self.fasilitas_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Calculate button
        calc_btn = ttk.Button(input_frame, text="Hitung Rekomendasi", command=self.show_recommendation)
        calc_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Recommendation result
        self.result_label = ttk.Label(self.root, text="", font=('Helvetica', 12))
        self.result_label.pack(pady=10)
        
        # Frame for data display
        data_frame = ttk.LabelFrame(self.root, text="Data Tempat Kos", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for data
        self.tree = ttk.Treeview(data_frame, columns=('nama', 'jarak', 'harga', 'fasilitas', 'alamat', 'rekomendasi'), show='headings')
        
        # Define headings
        self.tree.heading('nama', text='Nama Kos')
        self.tree.heading('jarak', text='Jarak (km)')
        self.tree.heading('harga', text='Harga (Rp)')
        self.tree.heading('fasilitas', text='Fasilitas (1-10)')
        self.tree.heading('alamat', text='Alamat')
        self.tree.heading('rekomendasi', text='Rekomendasi')
        
        # Set column widths
        self.tree.column('nama', width=120)
        self.tree.column('jarak', width=80)
        self.tree.column('harga', width=120)
        self.tree.column('fasilitas', width=100)
        self.tree.column('alamat', width=150)
        self.tree.column('rekomendasi', width=120)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Populate tree with data
        self.populate_treeview()
        
        # Button to show membership functions
        show_mf_btn = ttk.Button(self.root, text="Tampilkan Fungsi Keanggotaan", command=self.show_membership_functions)
        show_mf_btn.pack(pady=5)
        
        # Button to show outliers
        show_outliers_btn = ttk.Button(self.root, text="Tampilkan Outlier", command=self.show_outliers)
        show_outliers_btn.pack(pady=5)
    
    def populate_treeview(self):
        """Populate treeview with data and recommendations"""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Calculate recommendations and add to treeview
        for idx, row in self.kos_data.iterrows():
            score, rec_text = self.calculate_recommendation(row['jarak'], row['harga'], row['fasilitas'])
            
            # Format harga
            harga_text = f"Rp {row['harga']:,}"
            
            # Add to treeview
            self.tree.insert('', tk.END, values=(
                row['nama'],
                row['jarak'],
                harga_text,
                row['fasilitas'],
                row['alamat'],
                f"{rec_text} ({score:.1f})"
            ), tags=('outlier' if self.kos_data.loc[idx, 'outlier'] else 'normal'))
        
        # Configure tag colors
        self.tree.tag_configure('outlier', background='#ffdddd')
        self.tree.tag_configure('normal', background='')
    
    def show_recommendation(self):
        """Show recommendation for manual input"""
        try:
            jarak = float(self.jarak_entry.get())
            harga = float(self.harga_entry.get())
            fasilitas = float(self.fasilitas_entry.get())
            
            if not (0.1 <= jarak <= 20):
                messagebox.showerror("Error", "Jarak harus antara 0.1-20 km")
                return
            if not (350000 <= harga <= 5000000):
                messagebox.showerror("Error", "Harga harus antara Rp 350.000 - Rp 5.000.000")
                return
            if not (0 <= fasilitas <= 10):
                messagebox.showerror("Error", "Fasilitas harus antara 1-10")
                return
            
            score, recommendation = self.calculate_recommendation(jarak, harga, fasilitas)
            
            # Classify as outlier if needed
            is_outlier = (jarak > 10 or harga > 3000000 or fasilitas < 3)
            
            result_text = f"Hasil Rekomendasi: {recommendation}\nSkor: {score:.1f}"
            if is_outlier:
                result_text += "\n\nCatatan: Input termasuk kriteria outlier"
            
            self.result_label.config(text=result_text)
            
        except ValueError:
            messagebox.showerror("Error", "Input harus berupa angka")
    
    def show_membership_functions(self):
        """Show membership functions in a new window"""
        mf_window = tk.Toplevel(self.root)
        mf_window.title("Fungsi Keanggotaan")
        mf_window.geometry("800x600")
        
        fig = Figure(figsize=(8, 6), dpi=100)
        
        # Plot jarak membership functions
        ax1 = fig.add_subplot(311)
        ax1.set_title('Fungsi Keanggotaan Jarak')
        for label in ['dekat', 'sedang', 'jauh']:
            ax1.plot(self.jarak.universe, fuzz.interp_membership(self.jarak.universe, self.jarak[label].mf, self.jarak.universe), label=label)
        ax1.legend()
        ax1.set_ylabel('Derajat Keanggotaan')
        ax1.set_xlabel('Jarak (km)')
        
        # Plot harga membership functions
        ax2 = fig.add_subplot(312)
        ax2.set_title('Fungsi Keanggotaan Harga')
        for label in ['murah', 'sedang', 'mahal']:
            ax2.plot(self.harga.universe, fuzz.interp_membership(self.harga.universe, self.harga[label].mf, self.harga.universe), label=label)
        ax2.legend()
        ax2.set_ylabel('Derajat Keanggotaan')
        ax2.set_xlabel('Harga (Rp)')
        
        # Plot fasilitas membership functions
        ax3 = fig.add_subplot(313)
        ax3.set_title('Fungsi Keanggotaan Fasilitas')
        for label in ['minimal', 'cukup', 'lengkap']:
            ax3.plot(self.fasilitas.universe, fuzz.interp_membership(self.fasilitas.universe, self.fasilitas[label].mf, self.fasilitas.universe), label=label)
        ax3.legend()
        ax3.set_ylabel('Derajat Keanggotaan')
        ax3.set_xlabel('Tingkat Fasilitas (1-10)')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=mf_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_outliers(self):
        """Show only outliers in the data"""
        outlier_data = self.kos_data[self.kos_data['outlier']]
        
        if len(outlier_data) == 0:
            messagebox.showinfo("Outlier", "Tidak ditemukan data outlier")
            return
        
        # Create new window for outliers
        outlier_window = tk.Toplevel(self.root)
        outlier_window.title("Data Outlier")
        outlier_window.geometry("800x400")
        
        # Treeview for outliers
        tree = ttk.Treeview(outlier_window, columns=('nama', 'jarak', 'harga', 'fasilitas', 'alamat'), show='headings')
        
        # Define headings
        tree.heading('nama', text='Nama Kos')
        tree.heading('jarak', text='Jarak (km)')
        tree.heading('harga', text='Harga (Rp)')
        tree.heading('fasilitas', text='Fasilitas (1-10)')
        tree.heading('alamat', text='Alamat')
        
        # Set column widths
        tree.column('nama', width=150)
        tree.column('jarak', width=100)
        tree.column('harga', width=150)
        tree.column('fasilitas', width=100)
        tree.column('alamat', width=200)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(outlier_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Add data
        for _, row in outlier_data.iterrows():
            tree.insert('', tk.END, values=(
                row['nama'],
                row['jarak'],
                f"Rp {row['harga']:,}",
                row['fasilitas'],
                row['alamat']
            ))

if __name__ == "__main__":
    root = tk.Tk()
    app = KosRecommendationSystem(root)
    root.mainloop()