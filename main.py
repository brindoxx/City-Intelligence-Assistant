from dotenv import load_dotenv
load_dotenv()

import os
import requests
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient


# ============================================================
# ⚙️ PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="City Intelligence Assistant",
    page_icon="🏙️",
    layout="centered"
)


# ============================================================
# 🌤️ WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

    except Exception as e:
        return f"Unable to fetch weather data: {str(e)}"

    if str(data.get("cod")) != "200":
        return f"Unable to find weather for {city}."

    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    description = data["weather"][0]["description"]
    humidity = data["main"]["humidity"]

    return (
        f"The current weather in {city} is {description}, "
        f"with a temperature of {temp}°C. "
        f"It feels like {feels_like}°C and the humidity is {humidity}%."
    )


# ============================================================
# 📰 NEWS TOOL
# ============================================================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get the latest news for a given city."""

    try:
        response = tavily_client.search(
            query=f"{city} India news today",
            topic="news",
            search_depth="basic",
            max_results=3
        )

    except Exception as e:
        return f"Unable to fetch news: {str(e)}"

    results = response.get("results", [])

    if not results:
        return f"No recent news was found for {city}."

    news_list = []

    for r in results:

        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")

        news_list.append(
            f"📰 {title}\n"
            f"🔗 {url}\n"
            f"📝 {content[:250]}..."
        )

    return (
        f"Latest news for {city}:\n\n"
        + "\n\n".join(news_list)
    )


# ============================================================
# 🤖 GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# ============================================================
# 🛠️ TOOLS
# ============================================================

# bind_tools() expects a list
tools = [
    get_weather,
    get_news
]

llm_with_tools = llm.bind_tools(tools)


# Dictionary for executing tools
tools_by_name = {
    "get_weather": get_weather,
    "get_news": get_news
}


# ============================================================
# 🧠 SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "pending_tool_call" not in st.session_state:
    st.session_state.pending_tool_call = None


# ============================================================
# 🎨 SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏙️ City Intelligence")

    st.markdown(
        """
        ### 🤖 What can I do?

        Ask me things like:

        🌤️ **Weather**
        - What's the weather in Mumbai?
        - Is it raining in Delhi?

        📰 **News**
        - What's the latest news in Mumbai?
        - Give me today's news from Delhi.

        💬 **General Questions**
        - Tell me about Mumbai
        - What is the capital of Maharashtra?

        ---

        🔐 **Tool Safety**

        I'll ask for your permission before
        accessing external tools such as
        weather and news APIs.
        """
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = []
        st.session_state.pending_tool_call = None

        st.rerun()


# ============================================================
# 🏠 MAIN HEADER
# ============================================================

st.title("🏙️ City Intelligence Assistant")

st.caption(
    "🤖 Powered by Gemini • 🌤️ OpenWeather • 📰 Tavily"
)


# ============================================================
# 💬 DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    if message["role"] == "user":

        with st.chat_message("user"):
            st.markdown(message["content"])


    # --------------------------------------------------------
    # ASSISTANT MESSAGE
    # --------------------------------------------------------

    elif message["role"] == "assistant":

        with st.chat_message("assistant"):
            st.markdown(message["content"])


# ============================================================
# 🔧 HANDLE PENDING TOOL CALL
# ============================================================

if st.session_state.pending_tool_call:

    tool_call = st.session_state.pending_tool_call

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    st.warning(
        f"🔧 The assistant wants to use **{tool_name}** "
        f"with the following input:\n\n"
        f"`{tool_args}`"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "✅ Allow",
            use_container_width=True
        ):

            # Execute the requested tool
            tool_result = tools_by_name[
                tool_name
            ].invoke(tool_args)

            # Add tool result to LangChain conversation
            st.session_state.messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call["id"]
            })

            # Clear pending tool
            st.session_state.pending_tool_call = None

            # Continue conversation
            st.rerun()


    with col2:

        if st.button(
            "❌ Deny",
            use_container_width=True
        ):

            # Tell Gemini that the user denied the tool
            denial_message = (
                f"The user denied permission to use "
                f"the {tool_name} tool. "
                f"Please answer the user's request without "
                f"using this tool."
            )

            st.session_state.messages.append({
                "role": "tool",
                "content": denial_message,
                "tool_call_id": tool_call["id"]
            })

            st.session_state.pending_tool_call = None

            st.rerun()


# ============================================================
# 👤 USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask me about a city..."
)


if user_input:

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    # --------------------------------------------------------
    # CONVERT TO LANGCHAIN MESSAGES
    # --------------------------------------------------------

    langchain_messages = []

    for message in st.session_state.messages:

        role = message["role"]

        if role == "user":

            langchain_messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif role == "assistant":

            # Assistant messages are already natural language
            langchain_messages.append(
                {
                    "role": "assistant",
                    "content": message["content"]
                }
            )

        elif role == "tool":

            langchain_messages.append(
                ToolMessage(
                    content=message["content"],
                    tool_call_id=message["tool_call_id"]
                )
            )


    # --------------------------------------------------------
    # 🤖 ASK GEMINI
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            try:

                result = llm_with_tools.invoke(
                    langchain_messages
                )

            except Exception as e:

                st.error(
                    f"❌ Something went wrong: {str(e)}"
                )

                st.stop()


        # ====================================================
        # 🔧 GEMINI REQUESTED A TOOL
        # ====================================================

        if result.tool_calls:

            tool_call = result.tool_calls[0]

            # Save the tool call for approval
            st.session_state.pending_tool_call = {
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call["id"]
            }

            # Save assistant's tool request internally
            st.session_state.messages.append({
                "role": "assistant",
                "content": "🔧 I need to access an external tool to answer this."
            })

            st.rerun()


        # ====================================================
        # 💬 NORMAL ASSISTANT RESPONSE
        # ====================================================

        else:

            # result.content may occasionally be a list of
            # content blocks, so convert it safely to text.

            if isinstance(result.content, str):

                answer = result.content

            elif isinstance(result.content, list):

                text_parts = []

                for block in result.content:

                    if isinstance(block, dict):

                        if block.get("type") == "text":
                            text_parts.append(
                                block.get("text", "")
                            )

                    elif isinstance(block, str):

                        text_parts.append(block)

                answer = "\n".join(text_parts)

            else:

                answer = str(result.content)


            # ------------------------------------------------
            # DISPLAY ONLY CLEAN TEXT
            # ------------------------------------------------

            st.markdown(answer)


            # ------------------------------------------------
            # SAVE CLEAN ANSWER
            # ------------------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })