# 이론과 실습을 겸비한 다이나믹 강화학습 - 비제이퍼블릭

이론과 실습을 겸비한 다이나믹 강화학습  저장소입니다. 

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
- [2장. 강화학습을 배우기 위한 사전지식]()
- [3장. 마르코프 환경과 벨만방정식의 이해]()
- [4장. 다이나믹프로그래밍에서 강화학습으로]()
- [5장. Q-Network]()
- [6장. Actor-Critic]()
- [7장. 알파고의 기본원리 - Monte Carlo Tree Search]()


## 코드오류 및 제보
complexhhs@gmail.com


# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!).  Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.
