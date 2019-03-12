import { QLearningAgent, IQLearningAgent } from 'ml-q-learning';

function sumActionsStats(actionsStatsA: number[], actionsStatsB: number[]): number[] {
  const actionsStats: number[] = actionsStatsA.reduce((actionsStats: number[], actionStatsA: number, index: number) => {
    const actionStatsB = actionsStatsB[index];
    actionsStats.push(actionStatsA + actionStatsB);
    return actionsStats;
  }, []);
  return actionsStats;
}

enum SelectedUpdate {
  A,
  B
}

function chooseRandomUpdate(): SelectedUpdate {
  return Math.random() >= 0.5 ? SelectedUpdate.A : SelectedUpdate.B;
}

export class DoubleQLearningAgent<TAction = any> extends QLearningAgent<TAction> implements IQLearningAgent {
  protected async chooseActionAlgorithm(stateSerialized: string): Promise<number> {
    const stateSerializedA = `A${stateSerialized}`;
    const stateSerializedB = `B${stateSerialized}`;
    await this.createStateIfNotExist(stateSerializedA);
    await this.createStateIfNotExist(stateSerializedB);
    const actionsStatsA: number[] = await this.memory.getState(stateSerializedA);
    const actionsStatsB: number[] = await this.memory.getState(stateSerializedB);
    const actionsStats: number[] = sumActionsStats(actionsStatsA, actionsStatsB);
    const actionIndex = await this.pickActionStrategy(actionsStats, this.episode);
    return actionIndex;
  }
  
  protected async learningAlgorithm(action: number, reward: number, stateSerialized: string, stateSerializedPrime: string): Promise<[string, number[]]> {
    const selectedUpdate: SelectedUpdate = chooseRandomUpdate();
    if (selectedUpdate === SelectedUpdate.A) {
      const stateSerializedAPrime = `A${stateSerializedPrime}`;
      const actionPrime = await this.greedyPickAction(stateSerializedAPrime);
      const stateSerializedA = `A${stateSerialized}`;
      const actionsStatsA: number[] = await this.memory.getState(stateSerializedA);
      const stateSerializedBPrime = `B${stateSerializedPrime}`;
      const actionsStatsBPrime: number[] = await this.memory.getState(stateSerializedBPrime);
      actionsStatsA[action] = actionsStatsA[action] + this.learningRate * (reward + (this.discountFactor * actionsStatsBPrime[actionPrime]) - actionsStatsA[action]);
      return [stateSerializedA, actionsStatsA];
    }
    if (selectedUpdate === SelectedUpdate.B) {
      const stateSerializedBPrime = `B${stateSerializedPrime}`;
      const actionPrime = await this.greedyPickAction(stateSerializedBPrime);
      const stateSerializedB = `B${stateSerialized}`;
      const actionsStatsB: number[] = await this.memory.getState(stateSerializedB);
      const stateSerializedAPrime = `A${stateSerializedPrime}`;
      const actionsStatsAPrime: number[] = await this.memory.getState(stateSerializedAPrime);
      actionsStatsB[action] = actionsStatsB[action] + this.learningRate * (reward + (this.discountFactor * actionsStatsAPrime[actionPrime]) - actionsStatsB[action]);
      return [stateSerializedB, actionsStatsB];
    }
    throw new Error('The learning algorithm did not return anything.');
  }

}

