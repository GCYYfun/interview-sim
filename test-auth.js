// 测试认证是否正常工作
import { auth } from './src/lib/server/auth.js';
import { env } from '$env/dynamic/private';

console.log('测试 Better Auth 配置...');
console.log('ORIGIN:', env.ORIGIN);
console.log('BETTER_AUTH_SECRET 长度:', env.BETTER_AUTH_SECRET?.length);
console.log('BETTER_AUTH_SECRET 存在:', !!env.BETTER_AUTH_SECRET);

// 尝试创建 session
try {
  const session = await auth.api.getSession({
    headers: {
      'cookie': 'test-cookie'
    }
  });
  console.log('Session 检查结果:', session ? '有 session' : '无 session');
} catch (error) {
  console.error('Session 检查失败:', error.message);
}

console.log('测试完成');