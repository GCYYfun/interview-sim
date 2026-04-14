import { Model, Context, ConsoleLogger } from '../src/index.js';

/**
 * MengLong SDK 测试脚本
 * 
 * 包含两部分：
 * 1. 基础连通性测试 (Simple Chat)
 * 2. 流式工具调用最小循环 (Stream + Tool Loop)
 * 
 * 运行方式:
 * npx tsx examples/test_api.ts
 */

// 初始化模型（建议在 .configs.toml 中配置好 API Key 或设置环境变量）
const model = new Model('menglong/deepseek-chat', {
  // 启用 ConsoleLogger 以观察完整的请求/响应链路
  logger: new ConsoleLogger(),
  
  // 注入工具定义
  tools: [{
    type: 'function',
    function: {
      name: 'get_weather',
      description: '获取指定城市的天气信息',
      parameters: {
        type: 'object',
        properties: {
          city: { type: 'string', description: '城市名称' }
        },
        required: ['city']
      }
    }
  }]
});

/**
 * 第一部分：最简单的连通性测试
 */
async function runConnectivityTest() {
  console.log('\n' + '='.repeat(20));
  console.log('PART 1: 连通性测试');
  console.log('='.repeat(20));
  
  try {
    const resp = await model.chat(['你好，请介绍一下你自己。']);
    console.log('\n[响应内容]:', resp.text);
    console.log('\n[Token 使用]:', resp.usage);
  } catch (err) {
    console.error('连通性测试失败:', err);
  }
}

/**
 * 第二部分：LLM (Stream) + Tool 的最小循环
 */
async function runToolLoopTest() {
  console.log('\n' + '='.repeat(20));
  console.log('PART 2: 流式工具循环测试');
  console.log('='.repeat(20));

  const ctx = new Context();
  ctx.user('北京的天气怎么样？');

  console.log('\n[第一阶段]: 正在请求工具调用 (Streaming)...');
  
  let actions = null;
  
  // 1. 发起流式请求
  for await (const chunk of model.streamChat(ctx)) {
    // 实时打印文本增量
    if (chunk.output?.delta?.text) {
      process.stdout.write(chunk.output.delta.text);
    }
    // 当流结束且识别到工具调用时，actions 会在最后一帧（或特定帧）返回
    if (chunk.output?.actions) {
      actions = chunk.output.actions;
    }
  }

  // 2. 如果识别到工具调用，进入反馈循环
  if (actions && actions.length > 0) {
    const action = actions[0];
    console.log(`\n\n[检测到工具调用]: ${action.name} (ID: ${action.id})`);
    
    // 模拟本地执行工具
    console.log(`[执行本地业务]: 正在查询 ${JSON.stringify(action.arguments)} ...`);
    const mockResult = '晴转多云，气温 15°C - 25°C，微风。';

    // 将助手发出的指令和工具执行的结果喂回上下文
    ctx.assistant(undefined, actions);         // 记录 Action
    ctx.tool(action.id!, mockResult, action.name); // 记录 Outcome

    console.log('\n[第二阶段]: 喂回结果并获取最终回答 (Streaming)...');
    
    // 3. 发起第二次流式请求获取最终总结
    for await (const chunk of model.streamChat(ctx)) {
      if (chunk.output?.delta?.text) {
        process.stdout.write(chunk.output.delta.text);
      }
    }
    console.log('\n\n[测试完成]: 最小工具循环运行成功。');
  } else {
    console.log('\n[预测]: 未能触发工具调用，请检查模型能力或 Prompt。');
  }
}

async function main() {
  console.log('🚀 开始 MengLong SDK 功能测试...');
  
  await runConnectivityTest();
  await runToolLoopTest();
  
  console.log('\n✨ 所有测试执行完毕。');
}

main().catch(console.error);
