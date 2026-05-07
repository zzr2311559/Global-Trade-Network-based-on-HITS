import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_and_save_scatter(csv_path, output_image_path, top_n_annotate=15):
    """
    读取算法结果 CSV，绘制 Hub vs Authority 散点图，并保存为图片。
    """
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 设置绘图风格 (使用 seaborn 的白底网格主题)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 9))
    
    # 提取坐标和国家名
    x = df['Hub Score (核心出口国)']
    y = df['Authority Score (核心进口国)']
    countries = df['Country']
    
    # 绘制基础散点
    sns.scatterplot(x=x, y=y, alpha=0.7, color='#2c3e50', s=120, edgecolor='white')
    
    # 自动找出需要打文字标签的核心国家
    top_hubs = df.nlargest(top_n_annotate, 'Hub Score (出)')['Country'].tolist()
    top_auths = df.nlargest(top_n_annotate, 'Authority Score (进)')['Country'].tolist()
    labels_to_annotate = set(top_hubs + top_auths)
    
    # 添加文字标签
    for i, country in enumerate(countries):
        if country in labels_to_annotate:
            plt.annotate(country, 
                         (x[i], y[i]),
                         xytext=(6, 6), 
                         textcoords='offset points',
                         fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            
    # 画象限十字虚线 (以平均值为中心)
    plt.axvline(x.mean(), color='#e74c3c', linestyle='--', alpha=0.6, label='Mean Hub Score')
    plt.axhline(y.mean(), color='#3498db', linestyle='--', alpha=0.6, label='Mean Auth Score')
    
    # 图表标题与装饰
    plt.title('HITS Algorithm: Country Roles in European Food Trade (2021)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hub Score (Export Dominance / 核心出口倾向)', fontsize=12, fontweight='bold')
    plt.ylabel('Authority Score (Import Dominance / 核心进口倾向)', fontsize=12, fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    
    # ================= 核心：保存图片 =================
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight') # dpi=300 保证学术论文级别的高清画质
    print(f"📊 可视化图片已成功保存至: {output_image_path}")
    
    # 弹窗显示图片 (如果你是在服务器或 SSH 上运行，可以注释掉这一行)
    plt.show()

if __name__ == "__main__":
    # 定义输入和输出路径
    data_file = "./result/hits_scores_2021.csv"
    img_file = "./result/hits_scatter_2021.png"
    
    # 防呆检查：确保数据文件存在
    if not os.path.exists(data_file):
        print(f"❌ 找不到数据文件 {data_file}！")
        print("💡 请先运行: python main.py")
    else:
        plot_and_save_scatter(data_file, img_file)