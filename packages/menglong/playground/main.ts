import { Model, ConsoleLogger } from '../src/index.js';

const sendBtn = document.getElementById('sendBtn') as HTMLButtonElement;
const output = document.getElementById('output') as HTMLDivElement;
const statusText = document.getElementById('statusText') as HTMLSpanElement;
const statusDot = document.getElementById('statusDot') as HTMLDivElement;

sendBtn.onclick = async () => {
    const apiKey = (document.getElementById('apiKey') as HTMLInputElement).value;
    const baseUrl = (document.getElementById('baseUrl') as HTMLInputElement).value;
    const modelId = (document.getElementById('model') as HTMLInputElement).value;
    const prompt = (document.getElementById('prompt') as HTMLInputElement).value;

    if (!apiKey) {
        alert('请输入 API Key');
        return;
    }

    // 更新 UI 状态
    output.innerText = '';
    statusText.innerText = '正在流式传输...';
    statusDot.classList.add('active');
    sendBtn.disabled = true;

    try {
        // 构建完整 Model ID
        // 如果用户没写前缀，为了匹配本地测试，默认补全 menglong/
        const finalModelId = modelId.includes('/') ? modelId : `menglong/${modelId}`;

        
        // 创建模型实例
        // 直接传入配置以绕过浏览器的 fs 限制
        const model = new Model(finalModelId, {
            config: {
                api_key: apiKey,
                base_url: baseUrl,
                timeout: 30
            },
            logger: new ConsoleLogger()
        });

        console.log(`[Playground] Requesting ${finalModelId}...`);

        let hasContent = false;
        
        // 使用针对浏览器的原生异步迭代器
        const stream = model.streamChat([prompt]);

        for await (const chunk of stream) {
            const delta = chunk.output?.delta?.text;
            if (delta) {
                if (!hasContent) {
                    output.innerText = ''; // 清除初始文字
                    hasContent = true;
                }
                output.innerText += delta;
                // 自动滚动到底部
                output.scrollTop = output.scrollHeight;
            }

            // 处理推理内容 (DeepSeek 特有)
            const reasoning = chunk.output?.delta?.reasoning;
            if (reasoning) {
                console.log('[Reasoning]:', reasoning);
            }
        }

        statusText.innerText = '请求成功';
    } catch (err: any) {
        statusText.innerText = '请求失败: ' + err.message;
        output.innerHTML = `<span style="color: #ef4444;">[Error]: ${err.message}</span>`;
        console.error('[Playground Error]:', err);
    } finally {
        statusDot.classList.remove('active');
        sendBtn.disabled = false;
    }
};
