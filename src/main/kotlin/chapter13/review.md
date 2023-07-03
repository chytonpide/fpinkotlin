# 13장 복기
## 학습목표
- 각 타입들의 관계에 대해서 이해하기
- 각 타입들이 어떤 역할을 가지는지 이해하기
- 레이피케이션이 어떤 흐름으로 이루어지는지 이해하기
- 레이피케이션을 통해서 어떤 개념들이 명시화되는지 이해하기

## 13 External effect and I/O

- 모나드는 값을 포장해서 순차적인 계산을 컴포지션 할 수 있게하는 효과를 제공한다.

## 13.1 Factoring effects out of an effectful program

- f:(A) -> B 인 순수하지 않은 함수가 있을때 우리는 이를 아래와 같은 두가지 파트로 나눌 수 있다.
    - (A) -> D 인 pure function, 이때 D 는 f 의 결과의 서술이다.
    - (D) -> B 인 impure function, 이때 이 impure function 을 D 의 interpreter 라 볼 수 있다.

## 13.2 Introducing the IO type to separate effectful code
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

### 13.2.1 Handling input effect

```kotlin
interface IO<A> {

    companion object {

        fun <A> unit(a: () -> A) = object : IO<A> {
            override fun run(): A = a()
        }

        operator fun <A> invoke(a: () -> A) = unit(a)
    }
 
    fun run(): A
 
    fun <B> map(f: (A) -> B): IO<B> =
        object : IO<B> {
            override fun run(): B = f(this@IO.run())
        }
 
    fun <B> flatMap(f: (A) -> IO<B>): IO<B> =
        object : IO<B> {
            override fun run(): B = f(this@IO.run()).run()
        }
 
    infix fun <B> assoc(io: IO<B>): IO<Pair<A, B>> =
        object : IO<Pair<A, B>> {
            override fun run(): Pair<A, B> =
                this@IO.run() to io.run()
        }
}
```
- 위 IO 타입은 run() 이 Unit 이 아닌 A 를 리턴한다.
- 위 IO 타입은 Monad 이다.
```kotlin
fun stdin(): IO<String> = IO { readLine().orEmpty() }
 
fun stdout(msg: String): IO<Unit> = IO { println(msg) }
 
fun converter(): IO<Unit> =
    stdout("Enter a temperature in degrees Fahrenheit: ").flatMap {
        stdin().map { it.toDouble() }.flatMap { df ->
            stdout("Degrees Celsius: ${fahrenheitToCelsius(df)}")
        }
    }
```
- 위의 converter 는 이팩트를 가지는 참조투명한 계산의 기술이다.

### 13.2.2 Benefits and drawbacks of the simple IO type
- 장점
  - IO computations are ordinary value. IO 계산을 value 로 만들었다. 이제 이 계산을 일반적인 값(저장하고 리스트에 넣고 하는)으로 다룰 수 있다.
  - IO computations 을 value 로 Reification 하면서 interpreter 를 분리해 낼 수 있었다. 우리는 다양한 interpreter 를 구현할 수 있고 이때 이의 클라이언트인 purecore 에는 아무런 영향을 끼치지 않는다.
    - 레이피케이션은 추상적인 개념을 더 구체적인 것으로 '표현' 하는 것이다. FP 에서는 개념을 타입으로 만든다.
- 단점
  - 지금의 구현으로는 stackoverflow 가 발생한다.
  - 지금의 구현은 monad 이기는 하지만, 어떤 연산을 lazy evaluation 할 수 있게 하는게 다 이다. 너무 일반적이어서 IO 값에 대해서 할 수 있는 추론이 매우 제한적이다.
  - 지금의 구현은 blocking IO action 만들 허용하고, 동시성이나 non blocking 에 대해 지원하지 않는다.

## 13.3 Avoiding stack overflow errors by reification and trampolining
```kotlin
val p: IO<Unit> =
    IO.monad()
        .forever<Unit, Unit>(stdout("Still going..."))
        .fix()
```

```kotlin
fun <B> flatMap(f: (A) -> IO<B>): IO<B> =
    object : IO<B> {
        override fun run(): B = f(this@IO.run()).run()
    }
```
- 위의 코드는 최초의 f로 돌아가야 하기 때문에 stack overflow 가 발생한다.

### 13.3.1 Reifying control flow as data constructors
- function call 에 대한 아무런 제약이 없는 program control flow 를 사용하지않고, 우리의 데이터 타입으로 명시적이고 바람직한 control flow 를 사용하는 것으로 이 문제를 해결할 수 있다.
  - interpreter 가 tail-recursive loop 를 가지도록 IO 데이터 타입의 constructor 인 FlatMap 을 정의할 수 있다.
```kotlin
sealed class IO<A> : IOOf<A> {
    companion object {
        fun <A> unit(a: A) = Suspend { a }
    }
 
    fun <B> flatMap(f: (A) -> IO<B>): IO<B> = FlatMap(this, f)
    fun <B> map(f: (A) -> B): IO<B> = flatMap { a -> Return(f(a)) }
}
 
data class Return<A>(val a: A) : IO<A>()
data class Suspend<A>(val resume: () -> A) : IO<A>()
data class FlatMap<A, B>(
    val sub: IO<A>,
    val f: (A) -> IO<B>
) : IO<B>()
```
- 이제 flatMap 은 무언가를 계한하는게 아니라 단순히 FlatMap data constructor 를 실행하도 컨트롤을 돌려준다.
- interpreter 가 FlatMap 을 만났을 때 sub 를 계산하고 그다음 f 를 실행하는것을 기억하게 할 수 있다. 
```kotlin
fun stdout(s: String): IO<Unit> = Suspend { println(s) }
 
val p = IO.monad()
    .forever<Unit, Unit>(stdout("To infinity and beyond!"))
    .fix()
```
- stdout 이 `IO { println(msg) }` 였는데 `Suspend { println(s) }` 로 바뀌었다. 
- Suspend 는 결과를 생성하기 위해 실행하야할 effect 를 의미한다.
- 위의 p 는 아래와 같이 표현할 수 있다.
```kotlin
FlatMap(Suspend{ println("To infinity and beyond!") }) { _ ->
    FlatMap(Suspend { println("To infinity and beyond!") }) { _ ->
        FlatMap(Suspend { println("To infinity and beyond!")}) { _ ->
            TODO("repeat forever...")
        }
    }
}
```

```kotlin
@Suppress("UNCHECKED_CAST")
tailrec fun <A> run(io: IO<A>): A =
    when (io) {
        is Return -> io.a
        is Suspend -> io.resume()
        is FlatMap<*, *> -> {
            val x = io.sub as IO<A>
            val f = io.f as (A) -> IO<A>
            when (x) {
                is Return ->
                    run(f(x.a))
                is Suspend ->
                    run(f(x.resume()))
                is FlatMap<*, *> -> {
                    val g = x.f as (A) -> IO<A>
                    val y = x.sub as IO<A>
                    run(y.flatMap { a: A -> g(a).flatMap(f) })
                }
            }
        }
    }
```
- 위의 interpreter 는 tail recursive 하다.
- 원래 flatMap 자체가 알맹이를 빼낸 다음, 같은 형태의 다음 엘리먼트를 생성하는 함수에 적용하는 것이다. f(this@IO.run()).run() 로는 stack 에 f 에 
대한 참조가 계속 쌓임으로 FlatMap 이라는 데이터 타입을 만들어서 그 데이터 타입이 f 를 기억하도록 하면 interpreter 를 tail recursive 하게 만들 수 
있는 것이다.
- run 이 IO 프로그램을 interpret 할때, 데이터 타입을 도입해서 프로그램의 컨트롤을 명시적으로 만들었다. 프로그램이 Suspend(s) 로 어떤 effect 의 실행을 원한다면
그대로 실행을 하도록 하고, FlatMap(x,f) 로 subroutine 의 호출을 원하는 경우에, call stack 을 사용하지 않도록 x() 를 실행시키고 그 다음 f 를 실행시킨다.
f 는 바로 Suspend, FlatMap, Return 중 하나를 돌려주고 다시 컨트롤을 run 에게 넘기도록했다.
- 프로그램을 실행시켰을 때 프로그램은 보이지 않는 곳에서 evaluation 되고 그 결과가 돌아오는 것이 아니라, 
Suspend 또는 FlatMap request 들을 만들고 다시 run 함수에 컨트롤을 넘겨준다. run 함수는 한번에 하나의 Suspend 만을 처리하고 이것이 반복된다.
이 run 함수를 trampoline 이라고 부르기도 한다.

### Trampolining: A general solution to stack overflow 
- stack overflow 가 발생하는 프로그램을 interpreter 가 trampolining 을 사용하는 IO 타입을 적용하면 stack overflow 없이 실행 시킬 수 있다.
- 그렇지만 interpreter 가 trampolining 을 사용하도록 하는 것은 IO 와 관련이 없음으로 trampolining 을 묘사하는 일반적인 타입을 정의해 보자.
```kotlin
val f = { x: Int -> Return(x) }
val g = List(100000) { idx -> f }
  .fold(f) { a: (Int) -> Tailrec<Int>, b: (Int) -> Tailrec<Int> ->
    { x: Int ->
      Suspend { Unit }.flatMap { _ -> a(x).flatMap(b) }
    }
  }
```
- 원래 stack overflow 가 발생하는 프로그램을 위처럼 표현할 수 있고
```kotlin
run(g(42))
```
- 위처럼 실행 시킬 수 있다. run 인터프리터를 사용해서 실행시킬 수 있다는 것에 주의하자.
- Tailrec 타입을 도입함으로써 이제 함수를 composition 할 때 일반적인 함수 composition 이 아닌 flatMap 을 사용하는 크라이슬리 composition 을 사용한다.


## 13.4 A more nuanced IO type
- trampolining 을 사용하면 stack overflow 문제를 해결할 수 있지만, effect 가 어떤 종류의 것인지 암시적인 문제와 병렬처리에 대한 문제는 아직 남아있다.
- interpreter 에서는 Suspend(s) 에서 s() 를 콜하는 것 밖에 할 수 없다. s 가 어떤 형태의 effect 를 가지는지 알 수 없다. 이러한 무지가 병렬처리 같은것을 할수 없게 한다.
```kotlin
sealed class Async<A> : AsyncOf<A> {
    fun <B> flatMap(f: (A) -> Async<B>): Async<B> =
        FlatMap(this, f)
 
    fun <B> map(f: (A) -> B): Async<B> =
        flatMap { a -> Return(f(a)) }
}
 
data class Return<A>(val a: A) : Async<A>()
data class Suspend<A>(val resume: Par<A>) : Async<A>()
data class FlatMap<A, B>(
    val sub: Async<A>,
    val f: (A) -> Async<B>
) : Async<B>()
```
- `data class Suspend<A>(val resume: () -> A) : IO<A>()` 가 `data class Suspend<A>(val resume: Par<A>) : Async<A>()`로 변경됨
- interpreter 도 A 를 리턴하는게 아니라 Par<A> 를 리턴하도록 변경됨. 
대칭성을 생각하면 Function<A> 를 사용할 때 A 를 돌려주니 Par<A> 를 사용할때는 Future<A> 를 돌려줘야 하지만 interpreter 를 사용한 결과도 결국 expression 이고
최종적으로 Par<A> 를 executor 와 함께 실행해야 Future<A> 를 얻을 수 있다.

```kotlin
@Suppress("UNCHECKED_CAST")
tailrec fun <A> step(async: Async<A>): Async<A> =
    when (async) {
        is FlatMap<*, *> -> {
            val y = async.sub as Async<A>
            val g = async.f as (A) -> Async<A>
            when (y) {
                is FlatMap<*, *> -> {
                    val x = y.sub as Async<A>
                    val f = y.f as (A) -> Async<A>
                    step(x.flatMap { a -> f(a).flatMap(g) })
                }
                is Return -> step(g(y.a))
                else -> async
            }
        }
        else -> async
    }
 
@Suppress("UNCHECKED_CAST")
fun <A> run(async: Async<A>): Par<A> =
    when (val stepped = step(async)) {
        is Return -> Par.unit(stepped.a)
        is Suspend -> stepped.resume
        is FlatMap<*, *> -> {
            val x = stepped.sub as Async<A>
            val f = stepped.f as (A) -> Async<A>
            when (x) {
                is Suspend -> x.resume.flatMap { a -> run(f(a)) }
                else -> throw RuntimeException(
                    "Impossible, step eliminates such cases"
                )
            }
        }
    }
```
- 왜 step 을 쓰는 걸까? 코드를 나누기 위함. recursive call 이 일어나는 경우는 결국 FlatMap(FlaMap ..) 의 형태이 이것들의 계산을 분리 해 낸다.    
```kotlin
@higherkind
sealed class Free<F, A> : FreeOf<F, A>
data class Return<F, A>(val a: A) : Free<F, A>()
data class Suspend<F, A>(val s: Kind<F, A>) : Free<F, A>()
data class FlatMap<F, A, B>(
    val s: Free<F, A>,
    val f: (A) -> Free<F, B>
) : Free<F, B>()
```
- Tailrec 과 Async 를 type constructor F 로 parameterize 한 Free 타입을 정의 했다. 


### 13.4.1 Reasonably priced monads
- free 라는건 F의 monad object 를 공짜로 만들 수 있다는 것이다. 당연한 이야기 이지만 F 는 monad 여야 한다. 
- ex13.1) Free 의 flatMap 을 정의.
  - Free 의 flatMap 은 단지 FlatMap 의 constructor 이다. 어떻게 evaluation 하는지는 interpreter 가 알고 있다. 
- ex13.2) Free<ForFunction0, A> 의 interpreter 인 `tailrec fun <A> runTrampoline(ffa: Free<ForFunction0, A>): A` 을 정의.
- ex13.3) Free<F, A> 의 interpreter 인 `fun <F, A> run(free: Free<F, A>, M: Monad<F>): Kind<F, A>` 을 정의. 
  - run 은 F의 monad 오브젝트가 필요하다. 
- Free<F, A> 의 의미
  - 본질적으로, 이는 값 타입 A를 제로 또는 그 이상의 여러 레이어의 F 로 포장한 재귀적인 구조이다.
  - 다른 한편으로, 이는 leaves 로 A 를, 가지는 F 로 묘사되는 트리이다.
  - 또 다른 한편으로, 이는 자유 변수 A 와 함께, F 에 의해 설명이 주어지는 언어로된 프로그램에 대한 추상적인 구문 트리이다.
- Free<F, A> 에서 flatMap 은 a 를 취한 뒤 여러 레이어의 F 를 만들 수 있음으로 이는 모나드이다.
- 결과를 얻기 위해서 구조의 대한 interpreter 는 모든 F 레이어들을 처리할 수 있어야 한다.
- 우리는 이 구조와 그것의 해석기를 상호작용하는 코루틴으로 볼 수 있으며, 타입 F는 이 상호작용의 프로토콜을 정의 한다. 따라서 F 를 선택 함으로써 어떤 종료의 상호작용을 허용할지 컨트롤 할 수 있다.


### 13.4.2 A monad that supports only console I/O
- Function0 는 가장 간단한 타입 파라미터 F 인 동시에, 무엇이 허용되는지에 대해서 가장 제한이 적은 타입이어서 Function0<A> 가 무엇을 할지 추론할 수 없다.
```kotlin
@higherkind
sealed class Console<A> : ConsoleOf<A> {
 
    abstract fun toPar(): Par<A>
 
    abstract fun toThunk(): () -> A
 
}
```
- Console<A> 의 toPar() 는 Console<A> 를 Par 로 해석한다. toThunk() 는 Function0 로 해석한다.
```kotlin
object ReadLine : Console<Option<String>>() {
 
    override fun toPar(): Par<Option<String>> = Par.unit(run())
 
    override fun toThunk(): () -> Option<String> = { run() }
 
    private fun run(): Option<String> =
        try {
            Some(readLine().orEmpty())
        } catch (e: Exception) {
            None
        }
}
 
data class PrintLine(val line: String) : Console<Unit>() {
 
    override fun toPar(): Par<Unit> = Par.lazyUnit { println(line) }
 
    override fun toThunk(): () -> Unit = { println(line) }
 
}
```
- F 의 선택으로써 console 과 상호작용 하는 위와 같은 algebraic data type 을 정의할 수 있다.
- Console<A> 는 A 를 산출하는 연산을 표현하고, ReadLine 과 PrintLine 의 형태만 가질수 있도록 제한되어 있다. (sealed class)
- 콘솔과의 상호작용만 하도록 제한된 IO 타입인 Console<A> 를 Free 의 F 로 선택 할 수 있다. 
```kotlin
typealias ConsoleIO<A> = Free<ForConsole, A>
companion object {
  fun stdin(): ConsoleIO<Option<String>> =
    Suspend(ReadLine)
  fun stdout(line: String): ConsoleIO<Unit> =
    Suspend(PrintLine(line))
}
```
- Free<ForConsole, A> 를 사용하면 console 과 상호작용하는 프로그램을 만들 수 있고, 우리는 손쉽게 Free<ForConsole, A> 가 다른종류의 IO 를 하지 
않을 것이라고 예상할 수 있다.
- Free<ForConsole, A> 을 run 하려면 Monad<Console> 이 필요하다. Console 은 Par 나 Function0 가 될 수 있는 타입임으로 Monad 가 될 수 없다.
- 대신에 Console 이 monadic 한 특정 타입(Par 나 Function0 같은)으로 변환 될가 될 수 있도록 Translate 타입을 정의하고 사용할 수 있다. (translate 를 reification 함) 
```kotlin
interface Translate<F, G> {
    operator fun <A> invoke(fa: Kind<F, A>): Kind<G, A>
}
 
fun consoleToFunction0() = object : Translate<ForConsole, ForFunction0> {
    override fun <A> invoke(
        fa: Kind<ForConsole, A>
    ): Kind<ForFunction0, A> =
        Function0(fa.fix().toThunk())
}

fun consoleToPar() = object : Translate<ForConsole, ForPar> {
    override fun <A> invoke(
        fa: Kind<ForConsole, A>
    ): Kind<ForPar, A> =
        fa.fix().toPar()
}
```
- 이렇게 Translate 인터페이스를 정의하면 run 을 아래와 같이 일반화 할 수 있다.
```kotlin
fun <F, G, A> runFree(
    free: Free<F, A>,
    t: Translate<F, G>,
    MG: Monad<G>
): Kind<G, A> =
    when (val stepped = step(free)) {
        is Return -> MG.unit(stepped.a)
        is Suspend -> t(stepped.resume)
        is FlatMap<*, *, *> -> {
            val sub = stepped.sub as Free<F, A>
            val f = stepped.f as (A) -> Free<F, A>
            when (sub) {
                is Suspend ->
                    MG.flatMap(t(sub.resume)) { a -> runFree(f(a), t, MG) }
                else -> throw RuntimeException(
                    "Impossible, step eliminates such cases"
                )
            }
        }
    }
```
- `fun <F, G, A> runFree(free: Free<F, A>, t: Translate<F, G>, MG: Monad<G>): Kind<G, A> `
```kotlin
fun <A> runConsoleFunction0(a: Free<ForConsole, A>): Function0<A> =
    runFree(a, consoleToFunction0(), functionMonad()).fix()
 
fun <A> runConsolePar(a: Free<ForConsole, A>): Par<A> =
    runFree(a, consoleToPar(), parMonad()).fix()
```
```kotlin
fun functionMonad() = object : Monad<ForFunction0> {
    override fun <A> unit(a: A): Function0Of<A> = Function0 { a }
    override fun <A, B> flatMap(
        fa: Function0Of<A>,
        f: (A) -> Function0Of<B>
    ): Function0Of<B> = { f(fa.fix().f()) }()
}
 
fun parMonad() = object : Monad<ForPar> {
    override fun <A> unit(a: A): ParOf<A> = Par.unit(a)
 
    override fun <A, B> flatMap(
        fa: ParOf<A>,
        f: (A) -> ParOf<B>
    ): ParOf<B> = fa.fix().flatMap { a -> f(a).fix() }
}
```
- ex13.4) Function0 의 flatMap 이 stack-safe 하지 않기때문에 runConsoleFunction0 도 stack-safe 하지 않다.
  `fun <A> runConsoleFunction0(a: Free<ForConsole, A>): Function0<A>` 
  `fun <A> runConsoleFunction0(a: Free<ForConsole, A>): A` 
  시그니처를 위처럼 바꿨다.
  Free 의 type parameters 를 변경할 수 있는 translate 를 정의했다.
  내부에서 Console 을 Function0 로 만드는 Translate 를 생성한 다음, translate 의 인수로 전달해 Free 의 type parameter 를 Function0 로 변경한 뒤
  stack-safe 한 runTrampoline 을 실행 하도록 했다.
- A value of type Free<F, A> is like a program written in an instruction set provided by F.
- Free 의 type parameter 는 Tailrec (Function0 가 effect), Async (Par 가 effect) 가 될 수 있다.
  - 더 효과를 추론할 수 있게 하기 위해서 Free 의 type parameter 로 Console(ReadLine 과 WriteLine 만 될 수 있는, kotlin 의 println() 과 readline()의 클라이언트인) 을 정의 했다.
  - Console 은 Par 나 Function0 가 될 수 있다. 앞으로도 이런 형태를 지원하기 위해서 Translate 인터페이스를 정의 했다.

### 13.4.3 Testing console I/O by using interpreters 
- `typealias ConsoleIO<A> = Free<ForConsole, A>` 
- ConsoleIO 타입에서 effect 가 일어나야 한다고 암시하는 것은 없다. 이것은 interpreter 의 책임으로 우리는 Console action 이 IO 를 수행하지 않는  
  순수한 값으로 translate 되도록 선택 할 수 있다. 예를 들어 testing 목적의 interpreter 는 IO 를 수행하지 않고 constant 를 돌려주도록 할 수 있다.
```kotlin
data class ConsoleReader<A>(val run: (String) -> A) : ConsoleReaderOf<A> {
 
    companion object
 
    fun <B> flatMap(f: (A) -> ConsoleReader<B>): ConsoleReader<B> =
        ConsoleReader { r -> f(run(r)).run(r) }
 
    fun <B> map(f: (A) -> B): ConsoleReader<B> =
        ConsoleReader { r -> f(run(r)) }
}
 
@extension
interface ConsoleReaderMonad : Monad<ForConsoleReader> {
 
    override fun <A> unit(a: A): ConsoleReaderOf<A> =
        ConsoleReader { a }
 
    override fun <A, B> flatMap(
        fa: ConsoleReaderOf<A>,
        f: (A) -> ConsoleReaderOf<B>
    ): ConsoleReaderOf<B> =
        fa.fix().flatMap { a -> f(a).fix() }
 
    override fun <A, B> map(
        fa: ConsoleReaderOf<A>,
        f: (A) -> B
    ): ConsoleReaderOf<B> =
        fa.fix().map(f)
}
```
- Console 이 ReaderMonad 가 될 수 있다는 것을 인터페이스에 명시한다.
```kotlin
sealed class Console<A> : ConsoleOf<A> {
  abstract fun toPar(): Par<A>

  abstract fun toThunk(): () -> A
  
  abstract fun toReader(): ConsoleReader<A>
}
```
- Console -> ConsoleReader 인 Translate 오브젝트를 정의함으로써, runFree 를 사용할 수 있다. 전용 interpreter 인 
  `fun <A> runConsoleReader(cio: ConsoleIO<A>): ConsoleReader<A>` 도 정의 할 수 있다. 
```kotlin
val consoleToConsoleReader =
    object : Translate<ForConsole, ForConsoleReader> {
        override fun <A> invoke(fa: ConsoleOf<A>): ConsoleReaderOf<A> =
            fa.fix().toReader()
    }
 
fun <A> runConsoleReader(cio: ConsoleIO<A>): ConsoleReader<A> =
    runFree(cio, consoleToConsoleReader, ConsoleReader.monad()).fix()
```
- 마찬가지로 IO 의 조금더 완전한 시뮬레이션을 위해서 하나는 input, 하나는 output 을 표현하는 buffered interpreter 를 만들 수 있다. 
```kotlin
data class Buffers(
    val input: List<String>,
    val output: List<String>
)
 
data class ConsoleState<A>(
    val run: (Buffers) -> Pair<A, Buffers>
) : ConsoleStateOf<A> {
    // implement flatMap and map here
}
 
@extension
interface ConsoleStateMonad : Monad<ForConsoleState> {
    // override unit and flatMap here
}
 
val consoleToConsoleState =
    object : Translate<ForConsole, ForConsoleState> {
        override fun <A> invoke(fa: ConsoleOf<A>): ConsoleStateOf<A> =
            fa.fix().toState()
    }
 
fun <A> runConsoleState(cio: ConsoleIO<A>): ConsoleState<A> =
    runFree(cio, consoleToConsoleState, ConsoleState.monad()).fix()
```
- 당연한 이야기 이지만 ConsoleState 는 모나드여야한다.
- Console 이 State 가 될 수 있다는 것을 인터페이스에 명시한다.
```kotlin
sealed class Console<A> : ConsoleOf<A> {
  abstract fun toPar(): Par<A>

  abstract fun toThunk(): () -> A
  
  abstract fun toReader(): ConsoleReader<A>

  abstract fun toState(): ConsoleState<A>
}
```
- Console 은 Console effect 는 것을 추론할수 있게 하는 Free 의 nuanced type parameter 이고,
  ReadLine, PrintLine 은 monad 보다 한단계 위에서 어떤 effect 인지 추론을 가능하게 해주는 타입이다.(instruction set 이다.)
  Console 의 구체적인 타입인 ConsoleReader 와 ConsoleState 는 monad 이고 테스트에 사용된다.
- runFree 는 generic 하고 우리는 ConsoleIO 프로그램이 실제로 side effect 를 가지는지 어떤지는 알 수 없다. 
  이는 우리가 어떤 interpreter 를 사용하는 지에 따라 결정된다. 프로그램은 referentially transparent expression 이다.
  이렇게 concern 이 분리되었다. 
   
## 13.5 Non-blocking and asynchronous I/O
- runConsolePar(p) 는 Par<Unit> 을 되돌려준다. 그래도 p 가 non-blocking 지원해야 실제로 non-blocking IO 가 가능하다.
- p 가 non-blocking I/O 를 지원하는 특정 라이브러리에 종속되게 할 수 없음으로 Source 인터페이스를 정의한 다음 p 또한 Par 를 사용한 expression 으로 만들어야 한다. 
  Par 는 monad 임으로 p 는 monadic compositional interface 로 만들어 진다.

```kotlin
interface Source {
  fun readBytes(
    numBytes: Int,
    callback: (Either<Throwable, Array<Byte>>) -> Unit
  ): Unit
}

val src: Source = TODO("define the source")
val prog: Free<ForPar, Unit> =
    readPar(src, 1024).flatMap { chunk1 ->
        readPar(src, 1024).map { chunk2 ->
            //do something with chunks
        }
    }
```

## 13.6 A general-purpose IO type
- IO 타입의 모든 단점을 해결 했음으로 이제 I/O를 수행하는 프로그램을 작성하는 일반적인 방법론을 개발할 수 있다.
- 우리는 우리의 프로그램을 작성하기 위해 Free<F, A> 를 생성할 수 있고, 이는 최종적으 낮은 레벨의 IO 타입(Async)으로 컴파일 된다.
- typealias IO<A> = Free<ForPar, A>
- 위 IO<A> 타입은 trampolined sequential execution 과 asynchronous execution 을 지원한다.
  우리의 메인 프로그램에서 우리는 각각의 효과를 가지는 타입을 이 일반적인 타입 아래로 가지고 올 수 있고, 
  프로그램을 실행시키기 위해서 필요한 것은 오직 F 를 Par 로 바꾸는 translation 이다.
- I/O를 수행하는 프로그램을 작성하는 일반적인 방법론론 이란, instruction set 을 명시 하는 F 를 정의하고, Par 로의 translation 을 정의하는 게 된다.
### 13.6.1 The main program at the end of the universe
- main 의 리턴값은 void 이다. 이것은 main side effect 를 가짐을 의미한다.
```kotlin
abstract class App {
 
    fun main(args: Array<String>) {
        val pool = Executors.newFixedThreadPool(8)
        unsafePerformIO(pureMain(args), pool)
    }
 
    private fun <A> unsafePerformIO(
        ioa: IO<A>,
        pool: ExecutorService
    ): A =
        run(ioa, Par.monad()).fix().run(pool).get()
 
    abstract fun pureMain(args: Array<String>): IO<Unit>
}
```
- 최종적으로 정의한 IO 는 typealias IO<A> = Free<ForPar, A> 임으로, 이를 실행시키기 위해서는 ExecutorService 가 필요하다.
  run(ioa, Par.monad()).fix() 는 Par 를 돌려주고 pool 을 사용해 실행하고 결과를 얻는다. 
- 우리의 프로그램은 effect 가 일어나는걸 관찰 할 수 없지만 그것이 일어나는 것은 알고 있다. 따라서 우리는 우리의 프로그램이 effect 를 가지고 있지만 side effect 는 없다고 말 할 수 있다.

## 13.7 Why the IO type is insufficient for streaming I/O
- IO monad 와 I/O action 을 first-class value 로 가질 수 있는 이점에도 불과하고, streaming I/O 는 monolithic loop 를 가지게 될 것이다.
- 화씨 온도가 쓰여진 fahrenheit.txt 를, 각 라인을 섭씨로 변경하여 celsius.txt 를 생성하는 프로그램을 작성한다면 아래와 같을 것 이다. 
```kotlin
@higherkind
interface Files<A> : FilesOf<A>
 
data class ReadLines(
    val file: String
) : Files<List<String>>
 
data class WriteLines(
    val file: String,
    val lines: List<String>
) : Files<Unit>
```

```kotlin
val p: Free<ForFiles, Unit> =
    Suspend(ReadLines("fahrenheit.txt"))
        .flatMap { lines: List<String> ->
            Suspend(WriteLines("celsius.txt", lines.map { s ->
                fahrenheitToCelsius(s.toDouble()).toString()
            }))
        }
```
- 위와 같은 프로그램을 잘 동작하지만 한번에 모든 파일을 메모리로 옮긴다음 프로그램이 실행되기 때문에 파일이 크다면 문제가 될 수 있다.  
- 상수 메모리만 사용해서 단계적으로 처리하도록 다음과 같은 API 를 가진 타입을 정의 할 수 있다. (I/O 핸들에 대한 접근을 제공하는 하위 레벨 파일 API 를 공개할 수 있습니다.)
```kotlin
@higherkind
interface FilesH<A> : FilesHOf<A>
 
data class OpenRead(val file: String) : FilesH<HandleR>
data class OpenWrite(val file: String) : FilesH<HandleW>
data class ReadLine(val h: HandleR) : FilesH<Option<String>>
data class WriteLine(val h: HandleW) : FilesH<Unit>
 
interface HandleR
interface HandleW
```
- 여는 것과 닫는 것을 따로 정의하고, ReadLine, WriteLine 은 인터페이스를 받는다.
- 이렇게 하는 것의 유일한 문제점은 아래와 같은 monolithic loop 를 가지는 코드를 작성해야 한다.
```kotlin
fun loop(f: HandleR, c: HandleW): Free<ForFilesH, Unit> =
    Suspend(ReadLine(f)).flatMap { line: Option<String> ->
        when (line) {
            is None ->
                Return(Unit)
            is Some ->
                Suspend(WriteLine(handleW {
                    fahrenheitToCelsius(line.get.toDouble())
                })).flatMap { _ -> loop(f, c) }
        }
    }
 
fun convertFiles() =
    Suspend(OpenRead("fahrenheit.txt")).flatMap { f ->
        Suspend(OpenWrite("celsius.txt")).map { c ->
            loop(f, c)
        }
    }
```
- Files 타입으로 파일을 열고 닫고, 실제 파일을 쓰고 읽는 것을 인터페이스로 정의한 한줄씩 읽고 쓰는 새로운 FilesH 타입으로 변경하여
  읽고 쓰는 인터페이스를 어떻게 구현하는지에 따라서 상수 메모리만 사용하는 효과적인 프로그램으로 만들었지만 프로그램은 위와 같은 monolithic loop 의 형태가 되고
  이 프로그램은 composable 하지 않다.
- 예를 들어 우리가 다루는 타입이 list 라면 아래와 같이 간단히 프로그램을 구성할 수 있다. (계산을 마치고 그 계산의 결과를 다시 movingAvg 에 넣음)
  - 위의 프로그램은 loop 를 수정하는 수 밖에 없다. (interpret 한 다음 그 결과를 사용한 계산 프로그램을 작성한다고 해도, 
    결국 constant 메모리만 사용하려면 loop 와 같은 함수를 만들고 그안에서 movingAvg 를 호출 해야 함)
```kotlin
fun movingAvg(n: Int, l: List<Double>): List<Double> = TODO()
 
val cs = movingAvg(
    5, lines.map { s ->
        fahrenheitToCelsius(s.toDouble())
    }).map { it.toString() }
```
- 요점은 List 와 같은 composable abstraction 을 사용한 programming 이 우리가 지금 구현한 primitive I/O operations 을 사용한 것보다 낫다는 것이다.
- IO 모나드는 외부 세계와의 상호작용을 표현하는 최소의 공통 분모이기 때문에 여전히 중요하다. IO 를 직접 사용하는 프로그램은 monolithic 해지는 경향이 있고 재사용에 
  제한이 있기 때문에 더 composable 하고 더 reusable 한 abstractions 을 발견할 것이다.


## 지금까지 컨택스트
### 13 External effects and I/O
#### 13.1: Factoring effects out of an effectful program
- 어떤 프로그램이든 pure core 랑 side effect 로 나눌 수 있다.
#### 13.2: Introducing the IO type to separate effectful code 
- 단순한 IO 타입을 도입, 실제 출력을 묘사만 하는 IO 타입을 도입해서 expression 과 실제 side effect 를 분리해 냄. 이 IO 타입은 monoid 이다.
- 13.2.1: Handling input effects
  - IO 타입이 입력을 다룰 수 있도록 변경, 이 IO 타입은 monad 이다.
  - IO 타입은 pure core 를 wrapping 하는 형태로 side effect 를 포함한 expression 을 만들 수 있다.
    - 다시 IO 타입은 monad 임으로 이 expression 들에 monadic law 를 적용할 수 있다. (Associativity 결합성, Right identity, Left identity) 즉 side effect 를 포함한 연산들을 조립 할 수 있다. 
    - 13.2.2 IO 타입으로 IO 의 계산을 일반적인 값으로 만들 수 있었지만, stackoverflow, 너무 일반적이어서 추론이 힘듬, non blocking 과 동시성을 지원하지 않는 문제가 있다.
### 13.3: Avoiding stack overflow errors by reification and trampolining  
- flatMap 에서 연산을 조합할 때 최초의 f 를 실행해야 함으로 stack 에 f 에 대한 참조가 쌓이게된다. 결과 stack overflow 가 발생한다.
- 13.3.1: Reifying control flow as data constructors
  - 연산을 flow 를 직접 control 할 수 있게 FlatMap 데이터 컨트스트럭터를 도입 tail recursive 한 interpreter 를 만들 수 있다.
- 13.3.2: Trampolining: A general solution to stack overflow 
  - 지금 까지 유추한 IO 타입으로 trampolining 을 묘사하는 타입인 Tailrec 을 정의 함.  
### 13.4 A more nuanced IO type
- interpreter 에서 Suspend 가 어떤 effect 를 가지는 알 수 없어서, parallelism 을 명시화 하는 Async 타입을 정의함.  
- Tailrec 과 Async 를 type constructor F로 parameterize 한 Free 타입을 정의 했다.
  - Tailrec 은 Function, Async 는 Par 가 계산해야 하는 effect 이다. 
- 13.4.1 Reasonably priced monads
  - Free<F, A> 의 interpreter 인 `fun <F, A> run(free: Free<F, A>, M: Monad<F>): Kind<F, A>` 를 정의
  - Free<F, A> 가 다음과 같은 의미를 가진다.
    - 이는 값 타입 A를 제로 또는 그 이상의 여러 레이어의 F 로 포장한 재귀적인 구조이다.
    - 이는 자유 변수 A 와 함께, F 에 의해 설명(instructions)이 주어지는 언어로된 프로그램에 대한 추상적인 구문 트리이다.
  - 우리는 이 구조와 그것의 해석기를 상호작용하는 코루틴으로 볼 수 있으며, 타입 F는 이 상호작용의 프로토콜을 정의 한다. 따라서 F 를 선택 함으로써 어떤 종료의 상호작용을 허용할지 컨트롤 할 수 있다.
- 13.4.2 A monad that supports only console I/O
  - Free 의 타입파라미터인 Function0 는 너무 일반적이여서 이것으로는 아무것도 추론할 수 없음으로 Console 을 정의했다.
  - Console 은 ReadLine 과 WriteLine 로만 구체화 될 수 있고 kotlin 의 println() 과 readline()를 사용한다.
  - Console 은 가장 일반적인 형태인 Function0 의 형태를 지원하고 또, 비동기 연산을 위해 Par 의 형태도 지원한다.
  - Console 자체는 monad 가 될 수 없으며, 모나드인 Function0 나 Par 형태로 변환 해야 interpret 할 수 있다.
  - 앞으로도 이러한 어떤 효과를 나타내는 타입을 다시 interpret 을 할수 있는 다양한 모나딕 타입으로 변환 하는 케이스를 지원하기 위해서 Translate 를 정의했다.
  - Free<F, A> 의 값은, 타입 파라미터 F 의 명령어 집합으로 쓰여진 프로그램으로 Console 에서는 이것이 ReadLine 과 WriteLine 이다.
  - 재귀적인 구조(Suspend) 와 모나딕 변수의 대입(FlatMap, Return) 은 Free 에 의해서 제공된다.
- 13.4.3 Testing console I/O by using interpreters
  - Test 를 위한, 실제로 side effect 를 가지지 않는 ConsoleReader 와 ConsoleState 를 정의했다. 
  - Console 은 Console effect 는 것을 추론할수 있게 하는 Free 의 nuanced type parameter 이고,
    ReadLine, PrintLine 은 monad 보다 한단계 위에서 어떤 effect 인지 추론을 가능하게 해주는 타입이다.(instruction set 이다.)
    Console 의 구체적인 타입인 ConsoleReader 와 ConsoleState 는 monad 이고 테스트에 사용된다.
  - runFree 는 generic 하고 우리는 ConsoleIO 프로그램이 실제로 side effect 를 가지는지 어떤지는 알 수 없다.
    이는 우리가 어떤 interpreter 를 사용하는 지에 따라 결정된다. 
    프로그램은 referentially transparent expression 이고, side effect 와 pure core 의 concern 은 완전히 분리 되었다. 
# 13.5 Non-blocking and asynchronous I/O
  - runConsolePar(p) 는 Par<Unit> 을 되돌려준다. 그래도 p 가 non-blocking 지원해야 실제로 non-blocking IO 가 가능하다.
  - p 가 non-blocking I/O를 지원하는 특정 라이브러리에 종속되게 할 수 없음으로 Source 인터페이스를 정의한 다음 p 또한 Par 를 사용한 expression 으로 만들어야 한다.
    Par 는 monad 임으로 p 는 monadic compositional interface 로 만들어 진다.
## 13.6 A general-purpose IO type
  - 우리는 우리의 프로그램을 작성하기 위해 Free<F, A> 를 생성할 수 있고, 이는 최종적으 낮은 레벨의 IO 타입(Async)으로 컴파일 된다.
  - typealias IO<A> = Free<ForPar, A> 는 trampolined sequential execution 과 asynchronous execution 을 지원함으로 우리가 원하는 최종형태의 IO 타입이라 할 수 있다. 
  - I/O를 수행하는 프로그램을 작성하는 일반적인 방법론론 이란, instruction set 을 명시 하는 F 를 정의하고, Par 로의 translation 을 정의하는 게 된다.
## 13.7 Why the IO type is insufficient for streaming I/O
  - File 타입을 정의하고 Free<ForFile, A> 를 리턴하는 온도의 단위 변환 프로그램을 작성함.
  - 계산할 때 file 전체를 메모리에 올림으로 정량적인 메모리만 사용하도록 file 에서 한줄만 읽는 FileH 인터페이스를 정의했지만, 프로그램을 변경할 때 loop expression 자체를 변경해야 함
  - 반면 list 와 같은 composable 한 타입은 expression 의 조합으로 간단하게 프로그램을 변경할 수 있음
  - IO 타입은 외부세계와의 상호작용을 표현하는 최소한의 공통 분모로써 중요하지만, composable 할 수 없다는 한계가 있기 때문에 개별 요소의 묶음을 추상화 하는 List 와 같이,
    IO 계산에서 사용될 수 있는 composable 한 abstractions 이 필요함



## 요약
- pure core 랑 side effect 로 나눌 수 있음, 실제 출력을 묘사만 하는 IO 타입을 도입해서 expression 과 실제 side effect 를 분리해 냄.
  IO 는 monad 여서 이 expression 연산들을 결합 할 수 있음. (컨택스트를 가지는 결합 연산을 만들 수 있음)
  연산을 stack-safe 하게 계산하기 위해서 reification 해서 각각 의미를 가지는 데이터 타입을 생성하는 컨스트럭터를 도입, Suspend 는 결과를 생성하기 위해 실행하야할 effect 를 의미한다.
  IO 타입에서 각각의 관심사로 Tailrec, Async 와 그 interpreter 를 정의, Free 를 정의 Free<Function0, A> 는 Tailrec, Free<Par, A> 는 Async,
  Monad 가 있으면 실행 가능한 Free 의 interpreter 정의.
  Console 정의, Free<Console, A> 는 ConsoleIO, Console 은 Monad 가 아님으로 ConsoleIO 는 그 자체로 interpreter 에 의해 해석될 수 없음으로,
  Console 을 Monad 인 특정 타입으로 변환 시키는 Translate 정의.
  Monad 와 Translate 가 있으면 실행 가능한 Free 의 interpreter 정의.
  ConsoleIO 가 Free<Function0, A> 로 해석 될 때 이 interpreter 는 stack-safe 하지 않음으로, Free<Function0, A> 로 변환 시킨다음
  interpreter 로 runTrampoline 을 사용하는 전용 interpreter 정의.
  모든 단점을 극복한 typealias IO<A> = Free<ForPar, A> 를 정의. 어떤 IO 를 수행하는지는 instruction set 을 명시 하는 F 에 의해 표현된다. 
  이 F 는 최종적으로 Par 로 translate 되어야 한다.
  IO 타입은 외부세계와의 상호작용을 표현하는 최소한의 공통 분모로써 중요하지만, composable 할 수 없다는 한계가 있기 때문에 개별 요소의 묶음을 추상화 하는 List 와 같이,
  IO 계산에서 사용될 수 있는 composable 한 abstractions 이 필요하다.