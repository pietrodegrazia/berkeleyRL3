# berkeleyRL3
Port para Python 3 do exercício de Aprendizado por Reforço de Berkeley. O material original, escrito em Python 2, está em http://ai.berkeley.edu/reinforcement.html.

O código foi portado do python 2 para 3 com o utilitário 2to3 e alguns ajustes manuais envolvendo tk e divisão inteira. Os testes manuais preliminares correram bem, mas alguns problemas ainda podem estar lá. Feedback e pull requests são bem-vindos! 

# Introdução

Neste projeto, você implementará iteração de valor e Q-learning. Você irá testar seus agentes primeiro no Gridworld (da aula) e, em seguida, aplicá-los a um controlador de robô simulado (Crawler) e Pacman.

Este projeto inclui um autograder para você verificar suas soluções em sua máquina. Ele pode ser executado em todas as questões com o comando: 

`python autograder.py`

Ele pode ser executado para uma questão em particular, como a q2, com:

`python autograder.py -q q2`

Ele pode ser executado para um teste em particular com comandos com a seguinte forma:

`python autograder.py -t test_cases/q2/1-bridge-grid`

Consulte o tutorial do autograder no Projeto 0 (http://ai.berkeley.edu/tutorial.html) para obter mais informações sobre como usar o autograder.

O código para este projeto contém os seguintes arquivos: 

**Arquivos que você vai editar:**

* valueIterationAgents.py: um agente de iteração de valor para resolver MDPs conhecidos.
* qlearningAgents.py: agentes Q-learning para o Gridworld, Crawler e Pacman.
* analysis.py: um arquivo para preencher com suas respostas às perguntas deste projeto.

**Arquivos que você deve ler, mas NÃO editar:**
* mdp.py   Define métodos gerais de MDPs.
* learningAgents.py   Define as classes base ValueEstimationAgent e QLearningAgent, que seus agentes irão estender.
* util.py   Utilitários, incluindo util.Counter, que é particularmente útil para Q-learners.
* gridworld.py   A implementação do Gridworld.
* featureExtractors.py   Classes para extrair recursos em pares (estado, ação). Usado para o agente Q-learning aproximado (em qlearningAgents.py). 

**Arquivos que você pode ignorar:**

* environment.py: Classe abstrata para ambientes gerais de aprendizagem por reforço. Usado por gridworld.py.
* graphicsGridworldDisplay.py: Exibição gráfica do Gridworld.
* graphicsUtils.py: utilitários gráficos.
* textGridworldDisplay.py: Plug-in para a interface de texto Gridworld.
* crawler.py: O código do crawler e o artefatos de teste. Você vai executá-lo, mas não vai editá-lo.
* graphicsCrawlerDisplay.py: GUI para o robô rastreador.
* autograder.py: autograder do projeto.
* testParser.py: Parser de arquivos de solução e testes do autograder
* testClasses.py: Classes de teste gerais de autocorreção (autograding) 
* test_cases/: diretório contendo os casos de teste para cada questão
* reforcementTestClasses.py: Classes de teste de autograding específicas do Projeto 3 


**Arquivos para editar e enviar:** Você preencherá partes de valueIterationAgents.py, qlearningAgents.py e analysis.py durante o exercício. Você deve enviar esses arquivos com seu código e comentários. Não altere os outros arquivos nesta distribuição nem envie qualquer um de nossos arquivos originais que não sejam esses arquivos. 

**Avaliação:** o autograder será executado no seu código para correção técnica. Não altere os nomes de quaisquer funções ou classes fornecidas dentro do código, ou você causará estragos no autograder. No entanto, a corretude de sua implementação - não os julgamentos do autograder - será o juiz final de sua pontuação. Se necessário, revisaremos e avaliaremos os envios individualmente para garantir que você receba o devido crédito pelo seu trabalho. 

**Desonestidade Acadêmica:** iremos comparar seu código com outros envios para verificação de plágio. Nós descobriremos se você copiar o código de outra pessoa e enviá-lo com pequenas alterações. Os detectores de plágio são muito difíceis de enganar, então, por favor, não tente. Confiamos que todos vocês enviarão apenas seus próprios trabalhos; por favor, não nos decepcione. Se você fizer isso, buscaremos as consequências mais sérias que pudermos.

**Conseguindo ajuda:** Você não está sozinho(a)! Se você empacar em algo, entre em contato com a equipe do curso para obter ajuda. O discord, o fórum de discussão e demais recursos existem para sua ajuda; por favor, use-os. Caso não precise de atendimento ao vivo, avise-nos e agendaremos. Queremos que esses projetos sejam gratificantes e instrutivos, não frustrantes e desmoralizantes. Mas não saberemos quando ou como ajudar, a menos que você peça. 

**Discussão:** Por favor tome cuidado para não postar spoilers (trechos de código com a solução).

# MDPs

Para começar, execute o Gridworld no modo de controle manual, que usa as teclas de seta: 

`python gridworld.py -m`

Você verá o layout da aula com duas saídas. O ponto azul é o agente. Observe que quando você pressiona para cima, o agente só se move para o norte 80% do tempo. Assim é a vida de um agente do Gridworld!

Você pode controlar muitos aspectos da simulação. Uma lista completa de opções está disponível executando: 

`python gridworld.py -h`

O agente default se move aleatoriamente:

`python gridworld.py -g MazeGrid`

Você deve ver o agente aleatório se batendo pelo grid até que aconteça de achar uma saída. Essa é uma vida ruim para um agente de IA.

*Nota:* O MDP do Gridworld é tal que você deve primeiro entrar em um estado pré-terminal (as caixas duplas mostradas na GUI) e então realizar a ação especial de sair ('exit') antes que o episódio realmente termine (no verdadeiro estado do terminal chamado TERMINAL_STATE, que não é mostrado na GUI). Se você executar um episódio manualmente, seu retorno total pode ser menor do que o esperado, devido à taxa de desconto (-d para alterar; 0,9 por padrão).

Observe a saída do console que acompanha a saída gráfica (ou use -t para tudo em texto). Você será informado sobre cada transição que o agente experimenta (para desligar, use -q).

Como no Pacman, as posições são representadas por coordenadas cartesianas (x,y) e quaisquer matrizes são indexadas por [x][y], com 'norte' sendo a direção de aumento de y, etc. Por padrão, a maioria das transições receberá uma recompensa de zero, embora você possa alterar isso com a opção de recompensa por viver (-r). 

# Questão 1: Iteração de Valor

Escreva um agente de iteração de valor em `ValueIterationAgent`, que foi parcialmente especificado para você em `valueIterationAgents.py`. Seu agente de iteração de valor é um planejador offline, não um agente de aprendizado por reforço e, portanto, a opção de treinamento relevante é o número de iterações do algoritmo de iteração de valor que ele deve executar (opção -i) em sua fase de planejamento inicial. `ValueIterationAgent` usa um MDP na construtora e executa a iteração de valor para o número especificado de iterações antes de retornar.

A iteração de valor calcula estimativas de k passos dos valores ótimos, V_k. Além de executar a iteração de valor, implemente os seguintes métodos para `ValueIterationAgent` usando V_k. 

* `computeActionFromValues(state)` calcula a melhor ação de acordo com a função de valor fornecida por `self.values`. 
* `computeQValueFromValues(state, action)` retorna o valor-Q do par (estado, ação) dado pela função de valor em `self.values`. 

Essas quantidades são todas exibidas na GUI: os valores são números nos quadrados, os valores-Q são números nos triângulos (um para cada ação) e a política são as setas em cada quadrado. 

*Importante:* Use a versão em "batch" da iteração de valor, onde cada vetor V_k é calculado a partir de um vetor fixo V_{k-1} (como na aula), não a versão "online" onde um único vetor de pesos é atualizado "in place". Isso significa que quando o valor de um estado é atualizado na iteração k com base nos valores de seus estados sucessores, os valores dos sucessores usados no cálculo devem ser aqueles da iteração k-1 (mesmo se alguns dos estados sucessores já tivessem sido atualizado na iteração k). A diferença é discutida em Sutton & Barto no 6º parágrafo do capítulo 4.1. 

*Observação:* Uma política sintetizada a partir de valores em profundidade k (que refletem as próximas k recompensas) refletirá na verdade as próximas k + 1 recompensas (ou seja, você retorna π_{k+1}). Da mesma forma, os valores Q também refletirão uma recompensa a mais do que os valores (ou seja, você retorna Q_{k+1}).

Você deve retornar a política sintetizada π_{k+1}.

*Dica:* Use a classe `util.Counter` em` util.py`, que é um dicionário com valor padrão zero. Métodos como `totalCount` devem simplificar seu código. No entanto, tome cuidado com `argMax`: o argmax real que você deseja pode ser uma chave que não está no contador!

*Nota:* Certifique-se de lidar com o caso no qual um estado não tem ações disponíveis em um MDP (pense no que isso significa para recompensas futuras).

Para testar sua implementação, execute o autograder: 

`python autograder.py -q q1`

O comando a seguir carrega seu `ValueIterationAgent`, que irá computar uma política e executá-la 10 vezes. Pressione uma tecla para percorrer os valores, valores-Q e a simulação. Você deve descobrir que o valor do estado inicial (V(início), que pode ser lido na GUI) e a recompensa média resultante empírica (impressa após o término das 10 rodadas de execução) são bastante próximos. 

`python gridworld.py -a value -i 100 -k 10`

*Dica:* No BookGrid padrão, a iteração de valor em execução para 5 iterações deve fornecer esta saída: 

`python gridworld.py -a value -i 5`

![image](https://user-images.githubusercontent.com/5452322/115279534-3d547200-a11d-11eb-8a57-78de87c17847.png)

*Avaliação:* Seu agente de iteração de valor será avaliado em um grid novo. Verificaremos seus valores, valores Q e políticas após números fixos de iterações e na convergência (por exemplo, após 100 iterações). 

# Question 2: Análise da Travessia de Ponte

`BridgeGrid` é um mapa em grade com um estado terminal de baixa recompensa e um estado terminal de alta recompensa separados por uma "ponte" estreita, em cada lado da qual há um abismo de recompensa altamente negativa. O agente começa próximo ao estado de baixa recompensa. Com o desconto padrão de 0,9 e o ruído padrão de 0,2, a política ótima não cruza a ponte. Altere apenas UM dos parâmetros de desconto e ruído para que a política ótima faça com que o agente tente cruzar a ponte. Coloque sua resposta em `question2 ()` de `analysis.py`. (O ruído se refere à frequência com que um agente termina em um estado de sucessor não intencional quando executa uma ação.) O padrão corresponde 

`python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2`

![image](https://user-images.githubusercontent.com/5452322/115284686-5eb85c80-a123-11eb-8de7-f79e3433cbe5.png)

*Avaliação:* Verificaremos se você alterou apenas um dos parâmetros fornecidos e, com essa alteração, um agente de iteração de valor correto deve cruzar a ponte. Para verificar sua resposta, execute o autograder: 

`python autograder.py -q q2`

# Question 3: Políticas

Considere o layout `DiscountGrid`, mostrado abaixo. Este grid tem dois estados terminais com payoff positivo (na linha do meio), uma saída próxima com payoff +1 e uma saída distante com payoff +10. A linha inferior da grade consiste em estados terminais com retorno negativo (mostrado em vermelho); cada estado nesta região de "penhasco" tem retorno de -10. O estado inicial é o quadrado amarelo. Podemos distinguir entre dois tipos de caminhos: (1) caminhos que "arriscam o penhasco" e viajam perto da linha inferior da grade; esses caminhos são mais curtos, mas correm o risco de gerar um grande retorno negativo, e são representados pela seta vermelha na figura abaixo. (2) caminhos que "evitam o penhasco" e viajam ao longo da borda superior da grade. Esses caminhos são mais longos, mas têm menos probabilidade de gerar grandes resultados negativos. Esses caminhos são representados pela seta verde na figura abaixo. 

![image](https://user-images.githubusercontent.com/5452322/115284813-89a2b080-a123-11eb-8af6-5d4cf2da4f71.png)


Nesta questão, você escolherá as configurações dos parâmetros de desconto, ruído e recompensa vitalícia para este MDP para produzir políticas ideais de vários tipos diferentes. Sua configuração dos valores dos parâmetros para cada parte deve ter a propriedade de que, se o seu agente seguisse sua política ótima sem estar sujeito a nenhum ruído, ele exibiria o comportamento dado. Se um determinado comportamento não for alcançado para qualquer configuração dos parâmetros, afirme que a política é impossível retornando a string `'NOT POSSIBLE' (em inglês mesmo).

Aqui estão os tipos de política ideais que você deve tentar produzir:

* Prefira a saída próxima (+1), arriscando o penhasco (-10)
* Prefira a saída próxima (+1), mas evitando o penhasco (-10)
* Prefira a saída distante (+10), arriscando o penhasco (-10)
* Prefira a saída distante (+10), evitando o penhasco (-10)
* Evite as saídas e o penhasco (portanto, um episódio nunca deve terminar)

Para verificar suas respostas, execute o autograder: 

`python autograder.py -q q3`

Cada método de `question3a()` até `question3e()` no `analysis.py`. deve retornar uma tupla de 3 itens (desconto, ruído, recompensa por viver). 

*Observação:* Você pode verificar suas políticas na GUI. Por exemplo, usando uma resposta correta para 3 (a), a seta em (0,1) deve apontar para o leste, a seta em (1,1) também deve apontar para o leste, e a seta em (2,1) deve apontar para o norte .

*Observação:* Em algumas máquinas, você pode não ver uma seta. Nesse caso, pressione um botão no teclado para alternar para a exibição de valor-Q e calcule mentalmente a política tomando o argmax dos valores-Q disponíveis para cada estado.

*Avaliação:* Verificaremos se a política desejada é retornada em cada caso. 

# Question 4: Q-Learning

Observe que seu agente de iteração de valor não aprende realmente com a experiência. Em vez disso, ele considera seu modelo MDP para chegar a uma política completa antes de interagir com um ambiente real. Quando ele interage com o ambiente, ele simplesmente segue a política pré-computada (e.g. torna-se um agente reflexivo). Essa distinção pode ser sutil em um ambiente simulado como um Gridword, mas é muito importante no mundo real, onde o MDP real não está disponível.

Agora você escreverá um agente Q-learning, que faz muito pouco na construtora, mas aprende por tentativa e erro a partir de interações com o ambiente por meio de seu método `update (state, action, nextState, recompensa)`. Um esboço de um Q-learner é especificado em `QLearningAgent` em` qlearningAgents.py`, e você pode selecioná-lo com a opção '-a q'. Para esta questão, você deve implementar os métodos `update`,` computeValueFromQValues`, `getQValue` e` computeActionFromQValues`. 

*Nota:* Para `computeActionFromQValues`, você deve quebrar empates aleatoriamente para um melhor comportamento. A função `random.choice()` ajudará. Em um determinado estado, mesmo as ações que seu agente não viu antes têm um valor-Q, especificamente um valor-Q de zero, e se todas as ações que seu agente viu antes tiverem um valor-Q negativo, a ação não vista pode ser ótima.

*Importante:* Certifique-se de que em suas funções `computeValueFromQValues` e` computeActionFromQValues`, você só acessa valores-Q chamando `getQValue`. Esta abstração será útil para a questão 8 quando você sobrescrever `getQValue` para usar features dos pares estado-ação ao invés dos pares estado-ação diretamente.

Com a atualização do Q-learning implementada, você pode assistir ao seu Q-learner aprender sob controle manual, usando o teclado: 

`python gridworld.py -a q -k 5 -m`

Lembre-se de que `-k` controlará o número de episódios que seu agente aprenderá. Observe como o agente aprende sobre o estado em que estava, não aquele para o qual se move, e "deixa o aprendizado por onde passar". Dica: para ajudar na depuração, você pode desligar o ruído usando o parâmetro `--noise 0.0` (embora isso obviamente torne o Q-learning menos interessante). Se você direcionar o Pacman manualmente para o norte e depois para o leste ao longo do caminho ótimo para quatro episódios, deverá ver os seguintes valores Q: 

![image](https://user-images.githubusercontent.com/5452322/115285640-7cd28c80-a124-11eb-978a-74e6c3dfe9a5.png)


*Avaliação:* executaremos seu agente Q-learning e verificaremos se ele aprende os mesmos valores-Q e política de nossa implementação de referência quando cada um é apresentado com o mesmo conjunto de exemplos. Para avaliar sua implementação, execute o autograder: 

`python autograder.py -q q4`

# Question 5: Epsilon Greedy

Complete o seu agente Q-learning implementando a seleção de ação epsilon-greedy em `getAction`, o que significa que ele escolhe ações aleatórias em uma fração epsilon do tempo e segue seus melhores valores-Q atuais caso contrário. Observe que escolher uma ação aleatória pode resultar na escolha da melhor ação - ou seja, você não deve escolher uma ação aleatória somente entre as sub-ótimas, mas sim *qualquer* ação aleatória permitida. 

`python gridworld.py -a q -k 100 `

Seus valores-Q finais devem ser semelhantes aos de seu agente de iteração de valor, especialmente ao longo de caminhos bastante percorridos. No entanto, seus retornos médios serão menores do que os previstos pelos valores-Q por causa das ações aleatórias e da fase inicial de aprendizagem.

Você pode escolher um elemento de uma lista de maneira uniformemente aleatória chamando a função `random.choice`. Você pode simular uma variável binária com probabilidade `p` de sucesso usando` util.flipCoin(p) `, que retorna` True` com probabilidade `p` e` False` com probabilidade `1-p`. 

Para testar sua implementação, execute o autograder:

`python autograder.py -q q5`

Sem nenhum código adicional, agora você deve ser capaz de executar um robô rastejador (crawler) com Q-learning: 

`python crawler.py`

Se isso não funcionar, você provavelmente escreveu algum código muito específico para o problema `GridWorld` e deve torná-lo mais geral para todos os MDPs.

Isso invocará o robô rastejante usando seu Q-learner. Experimente os vários parâmetros de aprendizagem para ver como eles afetam as políticas e ações do agente. Observe que o delay é um parâmetro da simulação, enquanto a taxa de aprendizado e o epsilon são parâmetros de seu algoritmo de aprendizado e o fator de desconto é uma propriedade do ambiente. 



# Questão 6: Revisitando a Travessia de Ponte

Primeiro, treine um Q-learner completamente aleatório com a taxa de aprendizado padrão no `BridgeGrid` sem ruído por 50 episódios e observe se ele encontra a política ótima. 

`python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1`

Agora tente o mesmo experimento com um epsilon de 0. Existe um epsilon e uma taxa de aprendizado para os quais é altamente provável (maior que 99%) que a política ótima seja aprendida após 50 iterações? `question6 ()` em `analysis.py` deve retornar OU uma tupla de 2 itens de` (epsilon, taxa de aprendizagem)` OU a string `'NOT POSSIBLE'` se não houver nenhuma. Epsilon é controlado por `-e`, a taxa de aprendizagem por` -l`. 

*Observação:* Sua resposta não deve depender do mecanismo exato de desempate usado para escolher as ações. Isso significa que sua resposta deve estar correta, mesmo se, por exemplo, girarmos todo o mundo da ponte em 90 graus.

Para avaliar sua resposta, execute o autograder: 

`python autograder.py -q q6`


# Questão 7: Q-Learning e Pacman

É hora de jogar Pacman! O Pacman vai jogar em duas fases. Na primeira fase, *treinamento*, Pacman começará a aprender sobre os valores das posições e ações. Como leva muito tempo para aprender valores-Q precisos, mesmo para grids minúsculos, os jogos de treinamento do Pacman são executados em modo silencioso por padrão, sem display GUI (ou console). Assim que o treinamento de Pacman for concluído, ele entrará no modo de *teste*. Durante o teste, `self.epsilon` e` self.alpha` do Pacman serão ajustados para 0.0, efetivamente interrompendo o Q-learning e desabilitando a exploração, a fim de permitir que Pacman tire proveito de sua política aprendida. Os jogos de teste são mostrados na GUI por padrão. Sem quaisquer alterações de código, você deve ser capaz de executar o Pacman Q-learning para grids muito pequenos, como a seguir: 

`python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid `

Observe que `PacmanQAgent` já está definido para você em termos do` QLearningAgent` que você já escreveu. `PacmanQAgent` só é diferente por ter parâmetros de aprendizagem padrão que são mais eficazes para o problema Pacman (`epsilon = 0.05, alpha = 0.2, gamma = 0.8`). Você receberá crédito total por esta questão se o comando acima funcionar sem exceções e seu agente vencer pelo menos 80% das vezes. O autograder executará 100 jogos de teste após os 2.000 jogos de treinamento.

* Dica: * Se seu `QLearningAgent` funciona para` gridworld.py` e `crawler.py`, mas não parece estar aprendendo uma boa política para Pacman em` smallGrid`, pode ser porque seu `getAction` e / ou Os métodos `computeActionFromQValues` não consideram adequadamente, em alguns casos, ações não vistas. Em particular, porque ações não-vistas têm por definição um valor-Q de zero, se todas as ações que *foram* vistas têm valores Q negativos, uma ação não-vista pode ser ótima. Cuidado com a função argmax do `util.Counter`!

*Observação:* Para avaliar sua resposta, execute: 

`python autograder.py -q q7`

*Nota:* Se você quiser experimentar os parâmetros de aprendizagem, você pode usar a opção `-a`, por exemplo` -a epsilon=0.1,alpha=0.3,gamma=0.7`. Esses valores ficarão acessíveis como `self.epsilon, self.gamma` e` self.alpha` dentro do agente.

*Nota:* Embora um total de 2010 jogos sejam jogados, os primeiros 2.000 jogos não serão exibidos por causa da opção `-x 2000`, que designa os primeiros 2.000 jogos para treinamento (sem saída). Portanto, você só verá o Pacman jogar os últimos 10 desses jogos. O número de jogos de treinamento também é passado ao seu agente como a opção `numTraining`.

*Nota:* Se você quiser assistir a 10 jogos de treinamento para ver o que está acontecendo, use o comando: 

`python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10`

During training, you will see output every 100 games with statistics about how Pacman is faring. Epsilon is positive during training, so Pacman will play poorly even after having learned a good policy: this is because he occasionally makes a random exploratory move into a ghost. As a benchmark, it should take between 1,000 and 1400 games before Pacman's rewards for a 100 episode segment becomes positive, reflecting that he's started winning more than losing. By the end of training, it should remain positive and be fairly high (between 100 and 350).

Make sure you understand what is happening here: the MDP state is the *exact* board configuration facing Pacman, with the now complex transitions describing an entire ply of change to that state. The intermediate game configurations in which Pacman has moved but the ghosts have not replied are *not* MDP states, but are bundled in to the transitions.

Once Pacman is done training, he should win very reliably in test games (at least 90% of the time), since now he is exploiting his learned policy.

However, you will find that training the same agent on the seemingly simple `mediumGrid` does not work well. In our implementation, Pacman's average training rewards remain negative throughout training. At test time, he plays badly, probably losing all of his test games. Training will also take a long time, despite its ineffectiveness.

Pacman fails to win on larger layouts because each board configuration is a separate state with separate Q-values. He has no way to generalize that running into a ghost is bad for all positions. Obviously, this approach will not scale.



# Question 8: Approximate Q-Learning

Implement an approximate Q-learning agent that learns weights for features of states, where many states might share the same features. Write your implementation in `ApproximateQAgent` class in `qlearningAgents.py`, which is a subclass of `PacmanQAgent`.

*Note:* Approximate Q-learning assumes the existence of a feature function f(s,a) over state and action pairs, which yields a vector f_1(s,a) ... f_i(s,a) ... f_n(s,a) of feature values. We provide feature functions for you in `featureExtractors.py`. Feature vectors are `util.Counter` (like a dictionary) objects containing the non-zero pairs of features and values; all omitted features have value zero.

The approximate Q-function takes the following form

Q(s,a) = ∑ f_i(s,a) w_i

 where each weight w_i is associated with a particular feature f_i(s,a). In your code, you should implement the weight vector as a dictionary mapping features (which the feature extractors will return) to weight values. You will update your weight vectors similarly to how you updated Q-values:

δ = r + γ max Q(s', a') - Q(s,a)

w_i ← w_i + α . δ . f_i(s,a)


Note that the δ term is the same as in normal Q-learning, and r is the experienced reward.

By default, `ApproximateQAgent` uses the `IdentityExtractor`, which assigns a single feature to every `(state,action)` pair. With this feature extractor, your approximate Q-learning agent should work identically to `PacmanQAgent`. You can test this with the following command:

`python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid `

*Important:*`ApproximateQAgent` is a subclass of `QLearningAgent`, and it therefore shares several methods like `getAction`. Make sure that your methods in `QLearningAgent` call `getQValue` instead of accessing Q-values directly, so that when you override `getQValue` in your approximate agent, the new approximate q-values are used to compute actions.

Once you're confident that your approximate learner works correctly with the identity features, run your approximate Q-learning agent with our custom feature extractor, which can learn to win with ease:

`python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid `

Even much larger layouts should be no problem for your `ApproximateQAgent`. (*warning*: this may take a few minutes to train)

`python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic `

If you have no errors, your approximate Q-learning agent should win almost every time with these simple features, even with only 50 training games.

*Grading:* We will run your approximate Q-learning agent and check that it learns the same Q-values and feature weights as our reference implementation when each is presented with the same set of examples. To grade your implementation, run the autograder:

`python autograder.py -q q8`

*Congratulations! You have a learning Pacman agent!*

