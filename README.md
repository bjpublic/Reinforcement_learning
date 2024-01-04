# 강화 학습 이론&실습  - 비제이퍼블릭

강화 학습 이론&실습  저장소입니다. 

---

- 상용패키지
```
pandas==1.4.2
gym==0.10.5
torchvision==0.9.1
matplotlib==3.5.2
scikit-learn==0.24.0
```

## 상용패키지 설치방법
```
git clone https://github.com/bjpublic/Reinforcement_learning.git
cd Reinforcement_learning
pip install -r requirements.txt
```

## 파이토치 설치방법(conda 환경)
```
conda install pytorch==1.8.1 -c pytorch
```

## gym부수기능 설치
```
pip install gym[all]
conda install -c conda-forge pyglet
```


## jupyter notebook 커널 환경 업데이트
아래 커맨드를 수행한후 jupyter notebook 새로고침 실행 -> kernel -> RL_scratch 선택!
```
>>> conda activate RL_scratch
>>> pip install ipykernel
>>> python -m ipykernel install --user --name RL_scratch --display-name RL_scratch
```

## 컨텐츠
- [1장. 환경설정]()
- [2장. 강화학습을 배우기 위한 사전지식](https://github.com/bjpublic/Reinforcement_learning/tree/main/Chapter02)
- [3장. 마르코프 환경과 벨만방정식의 이해](https://github.com/bjpublic/Reinforcement_learning/tree/main/Chapter03)
- [4장. 다이나믹프로그래밍에서 강화학습으로](https://github.com/bjpublic/Reinforcement_learning/tree/main/Chapter04)
- [5장. Q-Network](https://github.com/bjpublic/Reinforcement_learning/tree/main/Chapter05)
- [6장. Actor-Critic](https://github.com/bjpublic/Reinforcement_learning/tree/main/Chapter06)
- [7장. 알파고의 기본원리 - Monte Carlo Tree Search](https://github.com/bjpublic/Reinforcement_learning/tree/main/Chapter07)


## 코드오류 및 제보
complexhhs@gmail.com


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.
