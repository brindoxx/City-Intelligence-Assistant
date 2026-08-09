from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print


# =========================
# WEATHER TOOL
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Unable to fetch weather data.')}"

    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]

    return (
        f"The current weather in {city} is "
        f"{description} with a temperature of {temp}°C."
    )


# =========================
# NEWS TOOL
# =========================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_news(city: str) -> str:
    """Get current news for a given city."""

    response = tavily_client.search(
        query=f"Latest news in {city}",
        topic="news",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}."

    news_list = []

    for r in results:

        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n"
            f"  🔗 {url}\n"
            f"  📝 {snippet[:100]}..."
        )

    return (
        f"Latest news in {city}:\n\n"
        + "\n\n".join(news_list)
    )


# =========================
# LLM
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# IMPORTANT:
# bind_tools() expects a list/sequence

tools = [
    get_weather,
    get_news
]

llm_with_tools = llm.bind_tools(tools)


# Dictionary for executing tools later

tools_by_name = {
    "get_weather": get_weather,
    "get_news": get_news
}


# =========================
# CHAT LOOP
# =========================

messages = []

print("City Intelligence Assistant")
print("Type 'exit' to quit the program.")


while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break

    messages.append(
        HumanMessage(content=user_input)
    )

    while True:

        # IMPORTANT:
        # use .invoke()
        result = llm_with_tools.invoke(messages)

        messages.append(result)

        # =========================
        # TOOL CALL
        # =========================

        if result.tool_calls:

            for tool_call in result.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call["args"]

                # Human-in-the-loop confirmation

                confirm = input(
                    f"\nThe model wants to use the tool "
                    f"'{tool_name}' with input: {tool_args}. "
                    f"Do you want to proceed? (yes/no): "
                )

                if confirm.lower() == "no":

                    print(
                        "Tool call denied. "
                        "The model will continue without using the tool."
                    )

                    break

                # Execute tool

                tool_result = tools_by_name[
                    tool_name
                ].invoke(tool_args)

                # Add tool result to conversation

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                )

            continue

        # =========================
        # NORMAL RESPONSE
        # =========================

        else:

            print(result.content)

            break