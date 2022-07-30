# 리플레이 버퍼 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py 에서 핵심 아이디어 채용

class ReplayBuffer(object):
    def __init__(self,size):
        '''
        리플레이 버퍼 초기 설정
        입력: 
            size - 리플레이 버퍼의 크기. 버퍼크기를 넘어가면 선입선출로 이전데이터를 메모리상에서 없앰
        '''
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        
    def __len__(self):
        return len(self._storage)
    
    def add(self,obs_t,action,reward,obs_tp,done):
        '''
        리플레이 버퍼에 데이터 추가
        입력:
            obs_t - 현재관찰(상태)
            action - 현재액션
            reward - 보상
            obs_tp - 다음관찰(상태')
            done - 종료여부
        '''
        data = (obs_t,action,reward,obs_tp,done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
    def _encode_sample(self,idxes):
        '''
        리플레이 버퍼로부터 샘플링
        입력:
            idxes - 메모리에서 추출할 데이터 인덱스 [batch]
        출력:
            샘플된 데이터
        '''
        obs_ts, actions, rewards, obs_tps, dones = [],[],[],[],[]
        for idx in idxes:
            obs_t, action, reward, obs_tp, done = self._storage[idx]
            obs_ts.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obs_tps.append(obs_tp)
            dones.append(done)
        return (
                np.array(obs_ts),
                np.array(actions),
                np.array(rewards),
                np.array(obs_tps),
                np.array(dones)
            )
    def sample(self,batch_size):
        '''
        self._encode_sample을 호출할 함수
        입력: 
            batch_size - 배치사이즈(int)
        출력 - self._encode_sample 참조
        '''
        idxes = np.random.choice(range(self.__len__),batch_size)
        return self._encode_sample(idxes)
