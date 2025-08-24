from crewai import Agent, Task, Crew, LLM
from textwrap import dedent
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Enhanced CSS with modern styling
st.markdown("""
<style>
:root {
  --primary-color: #4f46e5;
  --secondary-color: #6366f1;
  --text-color: #1e293b;
  --bg-color: #ffffff;
  --card-bg: #f8fafc;
  --border-color: #e2e8f0;
  --success-color: #10b981;
  --warning-color: #f59e0b;
  --error-color: #ef4444;
  --font-main: 'Segoe UI', system-ui, -apple-system, sans-serif;
  --font-mono: 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --primary-color: #6366f1;
    --secondary-color: #818cf8;
    --text-color: #f8fafc;
    --bg-color: #0f172a;
    --card-bg: #1e293b;
    --border-color: #334155;
  }
}

/* Base styles */
html, body {
  font-family: var(--font-main);
  color: var(--text-color);
  background-color: var(--bg-color);
}

/* Enhanced text area */
.stTextArea textarea {
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  padding: 12px !important;
  background-color: var(--card-bg) !important;
  color: var(--text-color) !important;
  transition: all 0.3s ease;
}

.stTextArea textarea:focus {
  border-color: var(--primary-color) !important;
  box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2) !important;
}

/* Improved buttons */
.stButton>button {
  border: none !important;
  background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
  color: white !important;
  padding: 10px 24px !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.stButton>button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
  opacity: 0.9 !important;
}

/* Badge styling */
.badge {
  display: inline-flex;
  padding: 6px 12px;
  border-radius: 12px;
  font-size: 14px;
  font-weight: 500;
  align-items: center;
  background-color: var(--card-bg);
  border: 1px solid var(--border-color);
  color: var(--text-color);
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.badge-icon {
  margin-right: 8px;
  font-size: 16px;
}

/* Section headers */
.section-header {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--primary-color);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
}

.section-header-icon {
  margin-right: 8px;
}

/* Result cards */
.result-card {
  background-color: var(--card-bg);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  border: 1px solid var(--border-color);
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* Spinner animation */
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.spinner {
  animation: spin 1s linear infinite;
  display: inline-block;
  margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# Set page config with enhanced metadata
st.set_page_config(
    page_title="Course to LinkedIn Post Generator",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Transform course completions into professional LinkedIn posts"
    }
)

# Constants
GROQ_MODEL_LIST = [
    'groq/llama-3.3-70b-versatile',
    'groq/Gemma2-9b-It',
    'groq/deepseek-r1-distill-llama-70b',
    'groq/llama3-70b-8192',
]

TONE_OPTIONS = [
    "Formal/Corporate",
    "Enthusiastic",
    "Humble-Brag",
    "Storytelling",
    "Technical",
    "Consultative",
    "Casual",
    "Motivational"
]

def get_tone_example(tone: str) -> str:
    """Returns example LinkedIn post for the selected tone."""
    examples = {
        "Formal/Corporate": (
            "I am pleased to announce my completion of the [Course Name] certification program "
            "offered by [Organization].\n\n"
            "Key competencies acquired:\n"
            "- Advanced [Skill 1] techniques\n"
            "- Professional [Skill 2] methodologies\n"
            "- Industry-standard [Skill 3] practices\n\n"
            "I look forward to applying these enhanced capabilities in my professional engagements.\n"
            "#ProfessionalDevelopment #ContinuingEducation #IndustryCertification"
        ),
        "Enthusiastic": (
            "🎉 Just completed the [Course Name] program! 🚀\n\n"
            "So excited to add these skills to my toolkit:\n"
            "- [Skill 1] (with hands-on projects!)\n"
            "- [Skill 2] (including the latest techniques)\n"
            "- [Skill 3] (certified by [Organization])\n\n"
            "Can't wait to put these into practice! Who else is working with these technologies?\n"
            "#Certified #ExcitedToLearn #NewSkills"
        ),
        # [Other tone examples remain unchanged...]
    }
    return examples.get(tone, 
        "🚀 Excited to share that I've completed [Course Name] through [Organization]!\n\n"
        "- Gained skills in [Skill 1]\n"
        "- Mastered techniques for [Skill 2]\n\n"
        "Looking forward to applying these in my work!\n"
        "#Certification #ProfessionalGrowth"
    )

def create_badge(content: str, icon: str) -> str:
    """Creates a styled badge component."""
    return f"""
    <div class="badge">
        <span class="badge-icon">{icon}</span>
        {content}
    </div>
    """

# Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="section-header"><span class="section-header-icon">⚙️</span>Configuration</div>', 
                unsafe_allow_html=True)
    
    model = st.selectbox('Select AI Model', GROQ_MODEL_LIST, help="Choose the AI model for post generation")
    api_key = st.text_input('Groq API Key', type="password", 
                          help="Enter your Groq API key to enable post generation")
    selected_tone = st.selectbox('Post Tone', TONE_OPTIONS, 
                               help="Select the tone/style for your LinkedIn post")
    
    # Add some additional helpful information
    st.markdown("---")
    st.markdown("**Need help?**")
    st.markdown("- Paste your full course description")
    st.markdown("- Select your preferred tone")
    st.markdown("- Click 'Generate' to create 3 post variations")

# Main Content
st.title("🎓 Course to LinkedIn Post Generator")
st.markdown("Transform your course completion into a professional LinkedIn post that showcases your achievements effectively.")

# Input Section
with st.container():
    st.markdown('<div class="section-header"><span class="section-header-icon">📝</span>Course Description</div>', 
                unsafe_allow_html=True)
    course_description = st.text_area(
        label="Paste your course description below",
        height=250,
        label_visibility="collapsed",
        placeholder="Paste the full description of the course you completed including any skills, technologies, or certifications mentioned..."
    )

# Generate Button and Results
if api_key:
    if st.button('✨ Generate LinkedIn Post', type="primary", key="generate"):
        if not course_description.strip():
            st.warning("Please enter a course description to generate a post")
        else:
            # Display selected configuration
            col1, col2 = st.columns(2)
            col1.markdown(create_badge(model, "🤖"), unsafe_allow_html=True)
            col2.markdown(create_badge(selected_tone, "✍️"), unsafe_allow_html=True)
            
            st.markdown('---')
            
            # Generate posts
            with st.spinner('Generating your professional LinkedIn posts...'):
                progress_bar = st.progress(0)
                results = []
                
                for i in range(3):
                    try:
                        # Update progress
                        progress_bar.progress((i + 1) * 33)
                        
                        # Initialize agents and crew
                        llm = LLM(model=model, api_key=api_key)
                        
                        course_analyzer_agent = Agent(
                            role="Course Details Extractor",
                            goal="Identify key skills, certifications, and course highlights",
                            backstory=(
                                "Expert at parsing educational content and extracting "
                                "valuable insights for professional development."
                            ),
                            verbose=True,
                            allow_delegation=False,
                            llm=llm
                        )
                        
                        linkedin_writer_agent = Agent(
                            role="LinkedIn Content Creator",
                            goal="Craft engaging LinkedIn posts about course completions",
                            backstory=(
                                "Specializes in transforming educational achievements into "
                                "professional social media posts that attract attention."
                            ),
                            verbose=True,
                            allow_delegation=False,
                            llm=llm
                        )
                        
                        extract_course_details = Task(
                            description=dedent(f"""\
                                Analyze this course description:
                                {course_description}
                                
                                Extract:
                                - Course name and certifying organization
                                - 3-5 key skills/technologies
                                - Notable details (duration, instructors)
                            """),
                            expected_output="Structured summary of course details",
                            agent=course_analyzer_agent,
                        )
                        
                        write_linkedin_post = Task(
                            description=dedent(f"""\
                                Create a {selected_tone}-tone LinkedIn post about this course.
                                Include:
                                1. Tone-appropriate opening
                                2. Bulleted key skills
                                3. Certifying body
                                4. Professional CTA
                                5. Relevant hashtags
                                
                                Tone example:
                                {get_tone_example(selected_tone)}
                            """),
                            expected_output=f"LinkedIn post in {selected_tone} tone",
                            agent=linkedin_writer_agent,
                            context=[extract_course_details]
                        )
                        
                        crew = Crew(
                            agents=[course_analyzer_agent, linkedin_writer_agent],
                            tasks=[extract_course_details, write_linkedin_post],
                            verbose=True,
                        )
                        
                        result = crew.kickoff(inputs={"course_description": course_description})
                        results.append(str(result))
                    
                    except Exception as e:
                        st.error(f"Error generating post #{i+1}: {str(e)}")
                        results.append(f"Could not generate post #{i+1}. Please try again.")
                
                # Display results
                progress_bar.empty()
                
                for i, post in enumerate(results, 1):
                    with st.expander(f'Post Variation #{i}', expanded=i==1):
                        # Create a container for each post with a unique key
                        post_container = st.container()
                        post_container.markdown(post)
                
                # LinkedIn action button
                st.link_button('🚀 Share on LinkedIn', 
                             url='https://www.linkedin.com/feed/',
                             help="Open LinkedIn to share your new post")
else:
    st.warning('Please enter your Groq API key to enable post generation')