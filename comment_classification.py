import joblib
import streamlit as st
import pandas as pd
import re
from preprocessing import preprocess
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Load the necessary files (emoji, teen, translation, wrong words, stopwords)
# Example of loading emoji and other dictionaries

# Emoji Dictionary
with open('emojicon.txt', 'r', encoding="utf8") as file:
    emoji_lst = file.read().split('\n')
    emoji_dict = {key: str(value) for key, value in (line.split('\t') for line in emoji_lst)}

# Teen code Dictionary
with open('teencode.txt', 'r', encoding="utf8") as file:
    teen_lst = file.read().split('\n')
    teen_dict = {key: str(value) for key, value in (line.split('\t') for line in teen_lst)}

# Wrong words list
with open('wrong-word.txt', 'r', encoding="utf8") as file:
    wrong_lst = file.read().split('\n')

# Stopwords list
with open('vietnamese-stopwords.txt', 'r', encoding="utf8") as file:
    stopwords_lst = file.read().split('\n')

# Load the classification model
# Load the pipeline or classifier that was trained earlier
model = joblib.load('model_pipeline.pkl')

# GUI setup using Streamlit
st.image('hasaki_banner.jpg', use_container_width=True)
st.title("Sentiment Analysis with Hasaki.vn")

# Sidebar menu
menu = ["Business Objective", "New Prediction", "Product Analysis"]
menu_choice = st.sidebar.selectbox('Menu', menu)

# Information in Sidebar
st.sidebar.write("""#### Thành viên thực hiện:
                 Vũ Trung Kiên & Trần Phương Mai""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                 Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 15/12/2024""")

# Tabs synchronized with Sidebar Menu

## Use the menu selection to highlight the corresponding tab
if menu_choice == "Business Objective":
    st.subheader("Business Objective")
    st.write("""HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên toàn quốc; và hiện đang là đối tác phân phối chiến lược tại thị trường Việt Nam của hàng loạt thương hiệu lớn.""")
    st.write("""Từ những đánh giá của khách hàng, vấn đề được đưa ra là làm sao để các nhãn hàng hiểu khách hàng rõ hơn, biết họ đánh giá gì về sản phẩm, từ đó có thể cải thiện chất lượng sản phẩm cũng như các dịch vụ đi kèm.""")
    st.image("sentiment.jpg")
    st.write("=> Problem/ Requirement: Xây dựng hệ thống dựa trên lịch sử những đánh giá của khách hàng đã có trước đó. Dữ liệu được thu thập từ phần bình luận và đánh giá của khách hàng ở Hasaki.vn.")
    st.write("=> Xây dựng mô hình dự đoán giúp Hasaki.vn và các công ty đối tác có thể biết được những phản hồi nhanh chóng của khách hàng về sản phẩm hay dịch vụ (tích cực, tiêu cực hay trung tính), điều này giúp họ cải thiện sản phẩm/ dịch vụ và làm hài lòng khách hàng.")

elif menu_choice == "New Prediction":
        st.subheader("Sentiment Analysis Predictor")
        
        # Add a informative description
        st.markdown("""
        🔍 **Predict Sentiment of Customer Reviews**
        - Analyze the emotional tone of text: Positive or Negative
        - Support for Vietnamese language comments
        - Works with single text or multiple comments via file upload
        """)
        
        # Add an example section
        with st.expander("💡 See Example"):
            st.markdown("""
            **Example Inputs:**
            - Positive: "Sản phẩm tuyệt vời, rất hài lòng!"
            - Negative: "Chất lượng kém, không như mô tả"
            """)
        
        input_type = st.radio("Choose Input Method", ["Input Text", "Upload File"], help="Select how you want to input your reviews")
        user_content = None

        if input_type == "Input Text":
            user_content = st.text_area(
                "Enter your content:", 
                placeholder="Nhập nhận xét của bạn...",
                help="Nhập một hoặc nhiều nhận xét để phân tích cảm xúc"
            )
            
            # Add some sample buttons
            st.markdown("#### Quick Examples:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🌞 Positive"):
                    user_content = "Sản phẩm tuyệt vời, rất hài lòng!"
            with col2:
                if st.button("🌧️ Negative"):
                    user_content = "Chất lượng kém, không như mô tả"

            if user_content.strip():
                user_content = [user_content]  # Convert to list for processing

        elif input_type == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload a CSV or TXT file", 
                type=["csv", "txt"],
                help="Upload a file with multiple reviews. Each line should be a separate review."
            )
            if uploaded_file:
                # Read file as raw text
                raw_text = uploaded_file.read().decode("utf-8")
                user_content = raw_text.splitlines()

        # Perform preprocessing and prediction if content exists
        if user_content:
            st.write("🔮 Processing your input...")
            processed_content = [
                preprocess(text, emoji_dict, teen_dict, wrong_lst, stopwords_lst)
                for text in user_content
            ]
            predictions = model.predict(processed_content)

            # Display results with color coding
            results_df = pd.DataFrame({"Original Text": user_content, "Prediction": predictions})
            
            # Color mapping for predictions
            def color_prediction(pred):
                if pred == '😄 Positive':
                    return 'background-color: #d4edda; color: #155724;'
                elif pred == '😞 Negative':
                    return 'background-color: #f8d7da; color: #721c24;'
            
            styled_df = results_df.style.apply(lambda x: [color_prediction(val) for val in x], axis=1)
            
            st.write("### 📊 Prediction Results")
            st.dataframe(styled_df, use_container_width=True)
            
            if input_type == "Upload File" and len(user_content) > 1:
                st.write("### 📈 Sentiment Distribution")
                sentiment_counts = results_df['Prediction'].value_counts()
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], ax=ax)
                ax.set_title('Overall Sentiment Distribution')
                st.pyplot(fig)

elif menu_choice == 'Product Analysis':
    def load_data():
        df_products = pd.read_csv('San_pham_full.csv')
        df_ratings = pd.read_csv('Danh_gia_full.csv')
        df_ratings_da_xu_ly = pd.read_csv('data_clean_2.csv')
        return df_products, df_ratings, df_ratings_da_xu_ly

    df_products, df_ratings, df_ratings_da_xu_ly = load_data()

    # Function to Display Product Information
    def hien_thi_san_pham(ma_sp):
        try:
            # Tìm sản phẩm theo mã
            san_pham = df_products[df_products['ma_san_pham'] == ma_sp].iloc[0]
            
            # Tạo DataFrame để hiển thị
            data = {
                'Thuộc tính': [
                    'Mã sản phẩm',
                    'Tên sản phẩm',
                    'Giá bán',
                    'Giá gốc',
                    'Phân loại',
                    'Mô tả',
                    'Điểm trung bình'
                ],
                'Giá trị': [
                    san_pham['ma_san_pham'],
                    san_pham['ten_san_pham'],
                    f"{san_pham['gia_ban']:,.0f} VNĐ",
                    f"{san_pham['gia_goc']:,.0f} VNĐ",
                    san_pham['phan_loai'],
                    san_pham['mo_ta'],
                    f"{san_pham['diem_trung_binh']:.1f}⭐"
                ]
            }
            
            # Tạo DataFrame và hiển thị bằng streamlit
            df_info = pd.DataFrame(data)
            st.dataframe(
                df_info,
                column_config={
                    "Thuộc tính": st.column_config.Column(
                        width="medium"
                    ),
                    "Giá trị": st.column_config.Column(
                        width="large"
                    )
                },
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

    # Function to Analyze Ratings
    def analyze_ratings(ma_sp):
        try:
            # Join hai bảng
            df_merged = pd.merge(df_products, df_ratings, on='ma_san_pham', how='left')
            
            # Lọc dữ liệu cho sản phẩm được chọn
            product_ratings = df_merged[df_merged['ma_san_pham'] == ma_sp]
            if len(product_ratings) == 0:
                st.error("❌ Không tìm thấy sản phẩm với mã này")
                return

            # Lấy thông tin sản phẩm
            ten_sp = product_ratings['ten_san_pham'].iloc[0]

            # Tính toán thống kê
            rating_counts = product_ratings['so_sao'].value_counts().sort_index(ascending=False)
            total_ratings = len(product_ratings)
            rating_percentages = (rating_counts / total_ratings * 100).round(1)

            # Plot biểu đồ
            fig, ax = plt.subplots(1, 2, figsize=(15, 8))

            # Cột
            ax[0].bar(rating_counts.index, rating_counts.values, color='#FFB636')
            ax[0].set_title(f'Phân bố số lượng đánh giá\n{ten_sp}')
            ax[0].set_xlabel('Số sao')
            ax[0].set_ylabel('Số lượng đánh giá')

            # Biểu đồ tròn
            ax[1].pie(rating_percentages.values, labels=[f'{i} sao ({p}%)' for i, p in zip(rating_percentages.index, rating_percentages.values)], autopct='%1.1f%%', colors=['#FFB636', '#FFC75F', '#FFD88C', '#FFE4B3', '#FFF1D7'])
            ax[1].set_title('Phân bố phần trăm đánh giá')

            st.pyplot(fig)

            # Hiển thị thông tin tổng quan dưới dạng bảng
            stats_data = {
                'Loại đánh giá': [f"{stars} sao ({'⭐' * int(stars)})" for stars in rating_counts.index],
                'Số lượng': rating_counts.values,
                'Phần trăm': [f"{percentage:.1f}%" for percentage in rating_percentages.values]
            }
            
            df_stats = pd.DataFrame(stats_data)
            
            st.write(f"### THỐNG KÊ ĐÁNH GIÁ SẢN PHẨM")
            st.write(f"Mã sản phẩm: {ma_sp}")
            st.write(f"Tên sản phẩm: {ten_sp}")
            
            st.dataframe(
                df_stats,
                column_config={
                    "Loại đánh giá": st.column_config.Column(
                        width="medium"
                    ),
                    "Số lượng": st.column_config.NumberColumn(
                        width="small",
                        format="%d"
                    ),
                    "Phần trăm": st.column_config.Column(
                        width="small"
                    )
                },
                hide_index=True
            )
             # Thêm phần hiển thị bình luận mẫu ở đây
            st.write("\n### 📝 Một số bình luận của khách hàng:")
            reviews_sample = product_ratings[['noi_dung_binh_luan', 'so_sao']].dropna().head(5)
            
            for _, review in reviews_sample.iterrows():
                with st.container():
                    st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin: 5px 0;">
                            <div style="color: #FFB636;">{"⭐" * int(review.so_sao)}</div>
                            <div style="margin-top: 5px;">{review.noi_dung_binh_luan}</div>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

    # Function to Create Sentiment Word Clouds
    def create_sentiment_wordclouds(ma_sp):
        try:
            # Merge data
            df_merged = pd.merge(df_products, df_ratings, on='ma_san_pham', how='left')
            df_merged_2 = pd.merge(df_merged.drop(columns=['noi_dung_binh_luan']), df_ratings_da_xu_ly, on='id', how='left')
            # Lọc dữ liệu cho sản phẩm được chọn
            product_data = df_merged_2[df_merged_2['ma_san_pham'] == ma_sp]
            if len(product_data) == 0:
                st.error("❌ Không tìm thấy sản phẩm với mã này")
                return

            ten_sp = product_data['ten_san_pham'].iloc[0]

            # Wordclouds for sentiment analysis
            sentiment_colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c'}
            fig, axes = plt.subplots(2, 1, figsize=(20, 10))

            for idx, sentiment in enumerate(sentiment_colors.keys()):
                sentiment_comments = product_data[product_data['sentiment'] == sentiment]
                
# Wordclouds for sentiment analysis
            sentiment_colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c'}
            fig, axes = plt.subplots(2, 1, figsize=(20, 10))

            for idx, sentiment in enumerate(sentiment_colors.keys()):
                sentiment_comments = product_data[product_data['sentiment'] == sentiment]
                
                if len(sentiment_comments) > 0:
                    # Kết hợp tất cả comment - đã sửa cách gọi hàm preprocess
                    text = ' '.join(sentiment_comments['noi_dung_binh_luan'].apply(
                        lambda x: preprocess(x, emoji_dict, teen_dict, wrong_lst, stopwords_lst)
                    ))
                    
                    # Tạo wordcloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white', 
                        max_words=100, 
                        color_func=lambda *args, **kwargs: sentiment_colors[sentiment]
                    ).generate(text)
                    
                    axes[idx].imshow(wordcloud, interpolation='bilinear')
                    axes[idx].axis('off')
                    axes[idx].set_title(f'Wordcloud cho đánh giá {sentiment.upper()}\n({len(sentiment_comments)} bình luận)', pad=20, size=15)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()  # Giải phóng bộ nhớ

            # Hiển thị thống kê sentiment dưới dạng bảng
            sentiment_counts = product_data['sentiment'].value_counts()
            total_comments = len(product_data)
            
            stats_data = {
                'Loại cảm xúc': list(sentiment_colors.keys()),
                'Số lượng': [sentiment_counts.get(sentiment, 0) for sentiment in sentiment_colors.keys()],
                'Phần trăm': [f"{(sentiment_counts.get(sentiment, 0)/total_comments*100):.1f}%" 
                             if total_comments > 0 else "0%" 
                             for sentiment in sentiment_colors.keys()]
            }
            
            df_stats = pd.DataFrame(stats_data)
            
            st.write(f"### THỐNG KÊ PHÂN TÍCH CẢM XÚC BÌNH LUẬN")
            st.write(f"Mã sản phẩm: {ma_sp}")
            st.write(f"Tên sản phẩm: {ten_sp}")
            st.write(f"Tổng số bình luận: {total_comments:,d}")  # Thêm format số
            
            st.dataframe(
                df_stats,
                column_config={
                    "Loại cảm xúc": st.column_config.Column(
                        width="medium"
                    ),
                    "Số lượng": st.column_config.NumberColumn(
                        width="small",
                        format="%d"
                    ),
                    "Phần trăm": st.column_config.Column(
                        width="small"
                    )
                },
                hide_index=True
            )

        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

    # # Streamlit UI
    # st.title("Ứng Dụng Phân Tích Sản Phẩm")
    
    # ma_san_pham_input = st.number_input("Nhập mã sản phẩm", min_value=0)
    
    # if ma_san_pham_input:
    #     st.subheader("📦 THÔNG TIN SẢN PHẨM 📦")
    #     hien_thi_san_pham(ma_san_pham_input)
        
    #     st.subheader("📊 PHÂN TÍCH ĐÁNH GIÁ SẢN PHẨM 📊")
    #     analyze_ratings(ma_san_pham_input)

    #     st.subheader("📊 PHÂN TÍCH CẢM XÚC BÌNH LUẬN SẢN PHẨM 📊")
    #     create_sentiment_wordclouds(ma_san_pham_input)
    # Streamlit UI
    st.title("Ứng Dụng Phân Tích Sản Phẩm")

    # Thêm search box
    search_keyword = st.text_input("🔍 Tìm kiếm sản phẩm", placeholder="Nhập tên sản phẩm (ví dụ: kem)")

    if search_keyword:
        # Tìm kiếm trong tên sản phẩm, không phân biệt hoa thường
        mask = df_products['ten_san_pham'].str.contains(search_keyword, case=False, na=False)
        search_results = df_products[mask].copy()
        
        if len(search_results) > 0:
            # Tạo cột hiển thị giá đã format
            search_results['gia_hien_thi'] = search_results['gia_ban'].apply(lambda x: f"{x:,.0f} VNĐ")
            
            # Hiển thị kết quả tìm kiếm
            st.write(f"Tìm thấy {len(search_results)} sản phẩm:")
            
            # Tạo DataFrame để hiển thị kết quả tìm kiếm
            display_df = search_results[['ma_san_pham', 'ten_san_pham', 'gia_hien_thi', 'diem_trung_binh']].copy()
            
            # Hiển thị kết quả với format
            st.dataframe(
                display_df,
                column_config={
                    "ma_san_pham": st.column_config.NumberColumn(
                        "Mã SP",
                        width="small"
                    ),
                    "ten_san_pham": st.column_config.Column(
                        "Tên sản phẩm",
                        width="large"
                    ),
                    "gia_hien_thi": st.column_config.Column(
                        "Giá bán",
                        width="medium"
                    ),
                    "diem_trung_binh": st.column_config.NumberColumn(
                        "Đánh giá ⭐",
                        width="small",
                        format="%.1f"
                    )
                },
                hide_index=True
            )
            
            # Cho phép chọn sản phẩm từ kết quả tìm kiếm
            ma_san_pham_list = search_results['ma_san_pham'].tolist()
            ten_san_pham_list = search_results['ten_san_pham'].tolist()
            
            # Tạo danh sách options với format "Mã SP - Tên SP"
            options = [f"{ma} - {ten}" for ma, ten in zip(ma_san_pham_list, ten_san_pham_list)]
            
            selected_product = st.selectbox(
                "Chọn sản phẩm để phân tích:",
                options=options,
                format_func=lambda x: x.split(" - ")[1]  # Chỉ hiển thị tên SP trong dropdown
            )
            
            if selected_product:
                # Lấy mã sản phẩm từ option đã chọn
                ma_san_pham_input = int(selected_product.split(" - ")[0])
                
                # Hiển thị các phân tích
                st.subheader("📦 THÔNG TIN SẢN PHẨM 📦")
                hien_thi_san_pham(ma_san_pham_input)
                
                st.subheader("📊 PHÂN TÍCH ĐÁNH GIÁ SẢN PHẨM 📊")
                analyze_ratings(ma_san_pham_input)

                st.subheader("📊 PHÂN TÍCH CẢM XÚC BÌNH LUẬN SẢN PHẨM 📊")
                create_sentiment_wordclouds(ma_san_pham_input)
        else:
            st.warning(f"Không tìm thấy sản phẩm nào với từ khóa: {search_keyword}")
    else:
        st.info("👆 Nhập từ khóa để tìm kiếm sản phẩm")
