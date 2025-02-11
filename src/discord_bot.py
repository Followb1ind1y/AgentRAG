import discord
import os
from agent import agent_executor, generate_summary, save_summary_to_pdf
from dotenv import load_dotenv

load_dotenv()

# Discord Client Setup
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower() == 'exit':
        await message.channel.send("👋 Goodbye! Generating summary...")
        
        # 生成聊天历史的总结
        summary = generate_summary(thread_id=str(message.author.id))
        await message.channel.send(f"Summary: {summary}")

        # 将总结保存为PDF
        save_summary_to_pdf(summary)
        await message.channel.send("✅ Summary saved as PDF!")
        return

    config = {"configurable": {"thread_id": str(message.author.id)}}
    # for event in agent_executor.stream(
    #         {"messages": [{"role": "user", "content": message.content}]},
    #         stream_mode="values",
    #         config=config):
    response = agent_executor.invoke(
        {"messages": [{"role": "user", "content": message.content}]},
        config=config,
    )
    # 获取消息内容并检查是否为空
    content = response["messages"][-1].content.strip() if hasattr(response["messages"][-1], "content") else ""
    # 如果内容不为空，且超过了Discord限制，则分割消息
    if content:
        max_length = 500
        while len(content) > max_length:
            await message.channel.send(content[:max_length])  # 发送消息的前4000字符
            content = content[max_length:]  # 更新剩余的消息内容

        # 发送剩余的部分
        if content:
            await message.channel.send(content)
    else:
        print("Empty message content, skipping send.")

def run_discord_bot():
    DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
    client.run(DISCORD_TOKEN)

if __name__ == "__main__":
    run_discord_bot()