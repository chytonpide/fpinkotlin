# 13장 복기
## 학습목표
- 각 타입들의 관계에 대해서 이해하기
- 각 타입들이 어떤 역할을 가지는지 이해하기
- 레이피케이션이 어떤 흐름으로 이루어지는지 이해하기
- 레이피케이션을 통해서 어떤 개념들이 명시화되고 추상화되는지 이해하

## 13 External effect and I/O

- 모나드는 값을 포장해서 순차적인 계산을 컴포지션 할 수 있게하는 효과를 제공한다.

## 13.1 Factoring effects out of an effectful program

- f:(A) -> B 인 순수하지 않은 함수가 있을때 우리는 이를 아래와 같은 두가지 파트로 나눌 수 있다.
    - (A) -> D 인 pure function, 이때 D 는 f 의 결과의 서술이다.
    - (D) -> B 인 impure function, 이때 이 impure function 을 D 의 interpreter 라 볼 수 있다.

- 13.2 Introducing the IO type to separate effectful code
```kotlin
    interface IO {
    fun run(): Unit
}

fun stdout(msg: String): IO =
object : IO {
override fun run(): Unit = println(msg)
}

fun contest(p1: Player, p2: Player): IO =
stdout(winnerMsg(winner(p1, p2)))
```
- IO 타입을 도입함. contest 는 IO 값을 리턴하지만 실제로 무언가를 실행하지는 않음.
- 우리는 contest가 효과를 가지고 있거나 효과를 생성한다고 말하지만, 실제로 side effect를 가지는 것은 IO의 해석기인 run 메서드 이다.
- run() 을 실행햇을 때 비로소 side effect 가 발생한다.
- who the winner is, winnerMsg to calculate what the resulting message should be, and stdout to indicate that the message should be printed to the console.  <- f 의 결과 서술
```kotlin
interface IO {
    companion object {
        fun empty(): IO = object : IO {
            override fun run(): Unit = Unit
        }
    }
 
    fun run(): Unit
 
    fun assoc(io: IO): IO = object : IO {
        override fun run() {
            this@IO.run()
            io.run()
        }
    }
}
```
- 위의 IO 는 monoid 이다.
    - 모노이드는 다음과 같은 특성을 갖는다.
        - 닫힌 연산: 임의의 두 요소를 이항 연산에 적용하면 결과는 모노이드 집합에 속합니다.
        - 결합성: 이항 연산은 연산의 순서에 관계없이 항상 동일한 결과를 반환합니다.
        - 항등원: 모노이드는 항등원을 가지며, 항등원과 어떤 요소를 이항 연산하더라도 그 요소는 그대로 유지됩니다.
- 프로그램이 IO 를 리턴하도록 만듬, IO 는 side effect 를 delay 시키지만, 위와 같은 대수적 구조를 가질 수 있다.
- 이렇게 되면 프로그램의 description 에 algebra 가 제공하는 것을 활용 할 수 있다.  