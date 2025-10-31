import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("student_data_adjusted_rounded.csv")

model = joblib.load("score_predictor.pkl")

st.set_page_config(page_title="学生成绩分析与预测系统", page_icon="", layout="wide")

page = st.sidebar.radio("🎓 导航菜单", ["项目介绍", "专业数据分析", "成绩预测"])

def load_data():
    df = pd.read_csv(r'student_data_adjusted_rounded.csv') 
    return df
df = load_data()

if page == "项目介绍":
    st.title("🎓 学生成绩分析与预测系统")
    st.markdown('***')
    c1, c2 = st.columns([2,1])
    with c1:
        st.header('🗒 项目概述')
        st.write('本项目是基于一个Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。')
        st.subheader('主要特点')
        st.markdown('⚫︎ 📊 **数据可视化**: 多维度展示学生学业数据')
        st.markdown('⚫︎ 🏷 **专业分析**： 按专业分类的详细统计分析')
        st.markdown('⚫︎ 🔮 **智能预测**: 基于机器学习模型的成绩预测')
        st.markdown(' ⚫︎ 💡 **学习建议**： 根据预测结果基于机器学习模型的成绩预测')
        st.write('⚫︎ 根据预测结果提供个性化反馈提供个性化反馈 ')

    
    with c2:
        images = ['1.png',
                     '2.png',
                     '3.png']
        captions = ['页面预览1','页面预览2','页面预览3']
        if 'a' not in st.session_state:
            st.session_state['a'] = 0

        def nextimg():
            st.session_state['a'] =(st.session_state['a']+1) % len(images)

        def next2mg():
            st.session_state['a'] =(st.session_state['a']-1) % len(images)

        st.image(images[st.session_state['a']], captions[st.session_state['a']])

        x1, x2 = st.columns(2)
        with x1:
            st.button('上一张', on_click=next2mg, use_container_width=True)

        with x2:
            st.button('下一张', on_click=nextimg, use_container_width=True)

    st.markdown('***')
    st.header('🚀 项目目标')
    a1, a2, a3 = st.columns(3)
    with a1:
        st.subheader('🎯 目标一')
        st.markdown('**分析影响因素**')
        st.write('⚫︎ 多维度展示学生学业数据')
        st.write('⚫︎ 按专业分类的详细统计分析')
        st.write('⚫︎ 基于机器学习模型的成绩预测')
        st.write('⚫︎ 根据预测结果提供个性化反馈 ')

    with a2:
        st.subheader('📊 目标二')
        st.markdown('**可视化展示**')
        st.write('⚫︎ 专业对比分析')
        st.write('⚫︎ 性别差异研究')
        st.write('⚫︎ 学习模式识别')

    with a3:
        st.subheader('🚩 目标三')
        st.markdown('**成绩预测**')
        st.write('⚫︎ 机器学习模型')
        st.write('⚫︎ 个性化预测')
        st.write('⚫︎ 及时干预预警')
        
    st.markdown('***')
    st.header('🧑‍🔧 技术架构')
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown('**前端构架**')
        python_code = 'Streamlit'
        st.code(python_code, language=None)

    with b2:
        st.markdown('**数据处理**')
        python_code = '''Python
NumPy'''
        st.code(python_code, language=None)

    with b3:
        st.markdown('**可视化**')
        python_code = '''Plotly
Matplotlib'''
        st.code(python_code, language=None)

    with b4:
        st.markdown('**机器学习**')
        python_code = 'Sklearn'
        st.code(python_code, language=None)
        

        
    
    

elif page == "专业数据分析":
    st.title("📊 专业数据分析")
    st.markdown('***')
    st.subheader('1.各专业男女性别比例')
    pd.set_option('display.unicode.east_asian_width', True)
    data = pd.read_csv("D:/streamlit_env/test/student_data_adjusted_rounded.csv",encoding='utf-8')
    d1, d2 = st.columns([2,1])
    with d1:
        gender_fig = px.histogram(df, x="专业", color="性别", barmode="group",
                                  title="各专业男女性别比例",
                                  labels={"专业": "专业", "count": "人数"})
        st.plotly_chart(gender_fig, use_container_width=True)
    with d2:
        gender_data = df.groupby(["专业", "性别"])["学号"].count().reset_index()
        gender_pivot = gender_data.pivot(index="专业", columns="性别", values="学号").fillna(0)
        st.subheader("性别比例数据")
        st.dataframe(gender_pivot, use_container_width=True)
    st.markdown("---")


    st.markdown('***')
    st.subheader('2.各专业学习指标对比')
    metrics = ["每周学习时长（小时）", "期中考试分数", "期末考试分数"]
    metric_df = df.groupby("专业")[metrics].mean().reset_index()
    e1, e2 = st.columns([2,1])
    with e1:
        metric_fig = go.Figure()
        for metric in metrics:
            metric_fig.add_trace(go.Scatter(x=metric_df["专业"], y=metric_df[metric], name=metric))
        metric_fig.update_layout(title="各专业学习指标趋势对比",
                                 xaxis_title="专业", yaxis_title="指标值")
        st.plotly_chart(metric_fig, use_container_width=True)
    with e2:
        st.subheader("详细数据")
        st.dataframe(metric_df, use_container_width=True)        
        

    st.markdown('***')
    st.subheader('3.各专业出勤率分析')
    f1, f2 = st.columns([2,1])
    with f1:
         attendance_fig = px.density_heatmap(df, x="专业", y="上课出勤率",
                                           title="各专业出勤率分布",
                                           labels={"专业": "专业", "上课出勤率": "出勤率"})
         st.plotly_chart(attendance_fig, use_container_width=True)
    with f2:
        attendance_rank = df.groupby("专业")["上课出勤率"].mean().reset_index().sort_values("上课出勤率", ascending=False)
        attendance_rank["排名"] = attendance_rank["上课出勤率"].rank(ascending=False).astype(int)
        st.subheader("出勤率排名")
        st.dataframe(attendance_rank[["排名", "专业", "上课出勤率"]], use_container_width=True)
    


    st.markdown('***')
    st.subheader('4.大数据管理专业专项分析')
    bd_major = df[df["专业"] == "大数据管理"]
    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均出勤率", f"{bd_major['上课出勤率'].mean():.1%}")
    with col2:
        st.metric("期中成绩", f"{bd_major['期中考试分数'].mean():.1f}分")
    with col3:
        st.metric("期末成绩", f"{bd_major['期末考试分数'].mean():.1f}分")
    with col4:
        st.metric("平均学习时长", f"{bd_major['每周学习时长（小时）'].mean():.1f}小时")
    # 图表
    col1, col2 = st.columns(2)
    with col1:
        bd_score_fig = px.histogram(bd_major, x="期末考试分数", title="大数据管理专业期末成绩分布")
        st.plotly_chart(bd_score_fig, use_container_width=True)
    with col2:
        bd_hours_fig = px.box(bd_major, y="每周学习时长（小时）", title="大数据管理专业学习时长分布")
        st.plotly_chart(bd_hours_fig, use_container_width=True)
        

else:
   st.title("⛳️ 期末成绩预测系统")
   st.markdown("通过输入学生的学习特征，系统将基于机器学习模型预测期末成绩并提供个性化建议")
   st.markdown("---")

   with st.form("predict_form", clear_on_submit=False):
       st.subheader("📓 学生学习信息录入")
       col1, col2 = st.columns(2)
       with col1:
            student_id = st.text_input("学号", placeholder="例如：2023001001")
            gender = st.selectbox("性别", ["男", "女"])
            major = st.selectbox("专业", df["专业"].unique())
            study_hours = st.slider(
                "每周学习时长（小时）",
                min_value=0.0, max_value=50.0, step=0.5, value=15.0,
                help="学生平均每周投入学习的时长"
                )
            with col2:
                attendance = st.slider(
                    "上课出勤率",
                    min_value=0.0, max_value=1.0, step=0.01, value=0.8,
                    help="实际出勤课时/应出勤课时"
                    )
                mid_score = st.slider(
                    "期中考试分数",
                    min_value=0.0, max_value=100.0, step=1.0, value=70.0
                    )
                homework_rate = st.slider(
                    "作业完成率",
                    min_value=0.0, max_value=1.0, step=0.01, value=0.9,
                    help="已完成作业数/总作业数"
                    )
            submit_btn = st.form_submit_button("🔍 预测期末成绩", type="primary")
            if submit_btn and student_id:
                X = [[study_hours, attendance, mid_score, homework_rate]]
                pred_score = model.predict(X)[0]
                pred_score = max(0, min(100, round(pred_score, 1))) 
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.subheader(f"🧑‍🎓 学号 {student_id} 的预测结果")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f'<div class="result-highlight">{pred_score} 分</div>', unsafe_allow_html=True)
                with col2:
                    st.progress(pred_score / 100) 
                    if pred_score >= 90:
                        st.success("💯 成绩等级：优秀")
                        st.image("6.png", width=200)
                    elif pred_score >= 80:
                        st.success("🤩 成绩等级：良好")
                        st.image("7.png",width=200)
                    elif pred_score >= 60:
                        st.info("🈴 成绩等级：合格")
                        st.image("5.png", width=200)
                    else:
                        st.warning("☹️ 成绩等级：待提高")
                        st.image("4.png", width=200)
                    st.markdown('<div class="advice-box">', unsafe_allow_html=True)
                    st.subheader("💡 学习建议")
                if attendance < 0.7:
                    st.markdown("- 建议提高出勤率，课堂互动对平时成绩影响显著")
                if homework_rate < 0.8:
                    st.markdown("- 需加强作业完成质量，作业是巩固和复习知识的关键")
                if study_hours < 10:
                    st.markdown("- 建议增加每周学习时长，保证足够的知识消化时间")
                if pred_score < 60:
                    st.markdown("- 可针对性复习期中考试薄弱环节，及时寻求老师帮助")
                else:
                    st.markdown("- 保持当前学习状态，建议定期总结知识体系，查漏补缺")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            elif submit_btn:
                pass

   
