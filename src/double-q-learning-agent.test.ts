import { DoubleQLearningAgent } from './double-q-learning-agent';
import '@tensorflow/tfjs-node';

describe('Q-Learning', () => {
  test('create Q-Learning agent', () => {
    new DoubleQLearningAgent(['1', 2]);
  });

  test('simple action test', async () => {
    const actionTrue = 'ActionTrue';
    const actionFalse = 'ActionFalse';
    const agent = new DoubleQLearningAgent([actionTrue, actionFalse]);
    const step = await agent.play('PlayActionFalse');
    agent.reward(step, step.action === actionFalse ? 1 : -1);
    const step2 = await agent.play('PlayActionTrue');
    agent.reward(step2, step2.action === actionTrue ? 1 : -1);
    const step3 = await agent.play('PlayActionFalse');
    agent.reward(step3, step3.action === actionFalse ? 1 : -1);
    const step4 = await agent.play('PlayActionTrue');
    agent.reward(step4, step4.action === actionTrue ? 1 : -1);
    const step5 = await agent.play('PlayActionFalse');
    agent.reward(step5, step5.action === actionFalse ? 1 : -1);
    const step6 = await agent.play('PlayActionTrue');
    agent.reward(step6, step6.action === actionTrue ? 1 : -1);
    const step7 = await agent.play('PlayActionFalse');
    agent.reward(step7, step7.action === actionFalse ? 1 : -1);
    const step8 = await agent.play('PlayActionTrue');
    agent.reward(step8, step8.action === actionTrue ? 1 : -1);
    await agent.learn();
    const step9 = await agent.play('PlayActionFalse');
    expect(step9.action).toBe(actionFalse);
    const step10 = await agent.play('PlayActionTrue');
    expect(step10.action).toBe(actionTrue);
  })
})
