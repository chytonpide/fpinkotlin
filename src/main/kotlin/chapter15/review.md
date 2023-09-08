# 15장 복기
## 학습목표
- 각 타입들의 관계에 대해서 이해하기
- 각 타입들이 어떤 역할을 가지는지 이해하기
- 레이피케이션이 어떤 흐름으로 이루어지는지 이해하기
- 레이피케이션을 통해서 어떤 개념들이 명시화되는지 이해하기

# 15 Stream processing and incremental I/O
- 이 장에서 배우는 것들
  - Shortcomings of imperative IO
  - Transformation using stream transducers
  - Building an extensible Process type
  - Single input and output processing with Source and Sink
  - Combining multiple input streams using Tee
- 이 장에서 외부세계와 상호작용 하는 프로그램을 high-level compositional style 로 작성 할 수 있도록 하는 방법을 배운다.

## 15.1 Problems with imperative I/O: An example
```kotlin
fun linesGt40k(fileName: String): IO<Boolean> = IO {
    val limit = 40000
    val src = File(fileName)
    val br = src.bufferedReader()
    try {
        var count = 0
        val lines = br.lineSequence().iterator()
        while (count <= limit && lines.hasNext()) {
            lines.next()
            count += 1
        }
        count > limit
    } finally {
        br.close()
    }
}
```
- bufferedReader 를 사용함으로 전체 파일이 메모리에 로드되지 않고 incremental 하게 처리하지만 다음과 같은 단점이 있다. 
  - 우리는 작업을 완료하고 file 을 닫아야 하는걸 기억하지 않으면 안된다.
  - 파일은 line 별로 처리하지만 일단 파일을 열어야 한다. 만약 프로그램이 여러 파일에서 라인을 읽어들인다고 한다면,
    운영체제에서 지원하는 한번에 열수 있는 파일의 개 수 때문에 프로그램이 죽어버릴 수 있다.
- 우리는 resource safe 한 프로그램을 원한다 그러나,  
  Using IO directly can be problematic because it means our programs are entirely responsible for ensuring their own resource safety, and we get no help from the compiler in making sure they do this.
- 높은 레벨의 algorithm 과 I/O concerns 이 엮어있다. 이것은 가독성을 떨어뜨릴 뿐만 아니라 composition 을 방해한다.
  만약 프로그램의 요구사항이 4만 라인 전에 각 라인의 연이은 첫글자가 abracadabra 가 되는 첫 라인의 index 를 구하는 프로그램이라면 우리가 작성한 코드는 
  매우 파악하기 힘들게 되며 결과물이 제대로 동작할지 의문스럽게 될 것이다. 
  일반적으로 IO monad 를 사용해서 효율적인 코드를 작성한다는 것은 monolithic loop 를 작성한다는것이고,monolithic loop 는 composable 하지 않다.
- 만약 우리가 Sequence<String> 형태의 line 을 가지게 된다면 코드는 다음과 같을 것이다.
```kotlin
lines.withIndex().exists { it.index >= 40000 }
```
- Sequence 를 사용하면 blank 를 걸러서 카운트를 하는 프로그램은 이미 존재하는 combinators 들을 사용해서 다음과 같이 구성된다.
```kotlin
lines.filter { it.trim().isNotBlank() }
    .withIndex()
    .exists { it.index >= 40000 }
```
- 연속된 각 라인의 첫글자의 합이 abracadabra 찾는 프로그램은 다음과 같이 구성된다.
```kotlin
lines.filter { it.trim().isNotBlank() }
    .take(40000)
    .map { it.first() }
    .joinToString("")
    .indexOf("abracadabra")
```
- 우리가 위와 같은 프로그램을 작성하고 싶어도 우리는 Sequence<String> 를 얻을 수 없다. 따라서 다음과 같이 function 을 사용해야 할 것이다.
```kotlin
fun lines(fileName: String): IO<Sequence<String>> =
    IO {
        val file = File(fileName)
        val br = file.bufferedReader()
        val end: String by lazy { // 마지막에 br 을 닫고 개행을 출력하는 함수
            br.close()
            System.lineSeparator()
        }
 
        sequence {
            yieldAll(br.lineSequence())
            yield(end)
        }
    }
```
- IO 가 품고있는(리턴하는) Sequence<String> 가 pure value 가 아니다.Sequence 에서는 elements 의 stream 이 강제(확정) 되었을 때 비로서
  file 을 읽는 side effect 를 실행한다. 
- 위와 같은 function 과 함께 Sequence<String> 을 사용한 프로그램을 구성한다고 해도 다음과 같은 문제들이 있다.
  - resource safe 하지 않다. lines 에서 정의 된 sequence 는 파일을 전부 다 읽은 다음 br.close()를 실행하기 때문에
    `exists` 와 같은 combinator 를 사용해 stream 을 조기 종료 할 수 있는데 그때 file 은 open 된 상태로 남는다.
  - file 이 닫힌 후에도 Sequence 를 다시 순회 것을 막을 방법이 없다. 만약 Sequene 가 memoizes 된다면, 메모리 사용의 문제가 생길 것이고,
    그렇지 않다면 file 이 이미 닫혀있기 때문에 IOException 가 발생할 것이다.
  - Sequence 의 elements 의 강제(확정) 시키는것은 I/O side effect 를 가지기 때문에 두개의 thread 가 sequence 를 순회하면 
    예상치 못한 행동이 야기된다.
  - 우리는 Sequence 가 나중에 어떻게 사용될지 알지 못한다. Sequence 적절하게 사용하기 위해서는 계속해서 비공식적인 다음과 같은 지식이 따라다녀야한다.
    우리는 일반적인 value 처럼 Sequence<String> 를 조작할 수 없다. 우리는 Sequence<String> 가 어디서 비롯된 것인지 알고 있어야한다.
    이것은 composition 에 좋지 않은데 composition 에서는 어떤 값의 타입 말고는 아무것도 알지 못해야 하기 때문이다.

## 15.2 Transforming streams with simple transducers
- A stream transducer specifies a transformation from one stream to another. 
- 스트림이라는 용어는 지연 생성되거나 외부 소스에 의해서 지원되는 sequence 를 언급하는 하는 꽤 일반적인 용어로 사용된다.
- Process lets us express stream transformations.
- Process<I, O> 는 단순한 (Stream<I>) -> Stream<O> 함수가 아니라, state machine 을 가지고 있고, 이 state machine 은 driver 와 함께
  전진해야 한다. 이 driver 라는 함수는 단계적으로 Process 와 input stream 을 동시에 소비한다.
- Process 는 세 가지 상태 중 하나에 있을 수 있으며, 각각은 드라이버에게 무언가를 알리는 신호이다.
  - Emit: driver 에게 input stream 의 head value 가 output stream 으로 emit 되어야 한다는걸 나타낸다.
  - Await: input stream 에게 값을 요청한다. driver 는 recv function 에 다음에 사용가능한 값을 전달해야 한다.
  - Halt: driver 에게 읽거나 output 으로 출력해야하는 엘리먼트가 없음을 나타낸다. 

```kotlin
operator fun invoke(si: Stream<I>): Stream<O> =
    when (this) {
        is Emit -> Cons({ this.head }, { this.tail(si) })
        is Await -> when (si) {
            is Cons -> {
              val p:Process<I, O> = this.recv(Some(si.head())) // recv 는 Process 를 리턴한다. 즉 recv 가 다음 Process 의 State (Emit, Await, Halt) 를 결정한다.  
              p(si.tail()) // 재귀적
            }
            is Empty -> {
              val p:Process<I, O> = this.recv(None)
              p(si) // 재귀적
            }
        }
        is Halt -> Stream.empty()
    }
```
- 주어진 `p: Process<I, O>` 와 `si: Stream<I>` 에 대해서, 표현 `p(si)` 는 `Stream<O>` 를 생성한다. 
- Process.invoke() 가 driver 다. 이 driver 에서는 Stream 을 input 으로 받지만, file 과 같은 다른 소스를 input 으로 하는 driver 를 만들 수도 있다.   

### 15.2.1 Combinators for building stream transducers
- liftOne: Process 에 어떤 function 을 태우는 function 이때 Process 의 구체적 타입은 Await 이다.
```kotlin
fun <I, O> liftOne(f: (I) -> O): Process<I, O> =
    Await { i: Option<I> ->
        when (i) {
            is Some -> Emit<I, O>(f(i.get))
            is None -> Halt<I, O>()
        }
    }
```
```kotlin
val p = liftOne<Int, Int> { it * 2 }
p(Stream.of(1, 2, 3, 4)).toList()
```
```
Await -> Emit<I, O>(f(i.get), Halt()) -> Emit(si.tail()) 재귀호출
Emit -> Cons({ this.head }, { this.tail(si) }) -> Halt(si) 재귀호출
```
- repeat: 이 combinator 는 그 Process 의 Halt 생성자를 재귀적인 호출로 바꾸면서, 영원히 반복시킨다.
```kotlin
fun repeat(): Process<I, O> {
    fun go(p: Process<I, O>): Process<I, O> =
        when (p) {
            is Halt -> go(this) // Restarts the process if it halts on its own
            is Await -> Await { i: Option<I> ->
                when (i) {
                    is None -> p.recv(None)
                    else -> go(p.recv(i))
                }
            }
            is Emit -> Emit(p.head, go(p.tail))
        }
    return go(this)
}
```
- lift: Process 에 어떤 function 을 태우는 function, 생성된 이 프로세스는 Stream 과 매핑된다. 
```kotlin
fun <I, O> lift(f: (I) -> O): Process<I, O> = liftOne(f).repeat()
```
- Emit(1).repeat()
```kotlin
go(Emit(1, Halt())) --> Emit(1, Emit(1, Emit(1, Emit(1, go(Halt()))))
```
- liftOne<Int, Int> { it * 2 }.repeat()
```kotlin
recv = { i: Option<I> ->
  when (i) {
    is Some -> Emit<I, O>(f(i.get))
    is None -> Halt<I, O>()
  }
}
```
```
go(Await { ... Emit( {it *2}(i), Halt()) }) --> Await { ... go(Emit( {it *2}(i), Halt()))}
go(Emit( {it *2}(i), Halt())) --> Emit({it *2}(i), go(Halt()))
go(Halt()) --> go(Await { ... Emit( {it *2}(i), Halt()) }) --> Await { ... go(Emit( {it *2}(i), Halt()))}
go(Emit( {it *2}(i), Halt())) --> Emit({it *2}(i), go(Halt()))
go(Halt()) --> go(Await { ... Emit( {it *2}(i), Halt()) }) --> Await { ... go(Emit( {it *2}(i), Halt()))}
----------
Await { ... Emit({it *2}(i), Await { ... Emit({it *2}(i), Await { ... Emit( {it *2}(i), Halt()) } ))}
```
- Await 의 repeat 은 input stream 이 끝날 때 같이 끝이난다.
- Emit(1).repeat() 으로 무한 stream 을 얻을 수 없다. Process 는 Stream 의 변환기 이기 때문이다. 무한 stream 을 얻으려면 아래와 같은 코드가 되어야한다.
```kotlin
val units = Stream.continually(Unit)
lift<Unit, Int> { _ -> 1 }(units)
```
- 반복적인 Process 를 생성하는 기본적인 패턴
  - function 을 Await 에 태운다.
  - Await 내의 recv 는 반복에 관심이 없다. input 을 체크한 후에 Emit 또는 Halt 를 리턴한다. (상태를 바꾼다.)
  - repeat() 컴비네이터를 호출한다. 
  - repeat 은 Await 과 함께 동작하며 input 이 있는경우 recv 의 실행결과를 go 의 인수로 넘겨 재귀적으로 호출한다. 
    이때 Await 이 Halt 를 리턴할 경우 원래의 Await 을 go 의 인수로 넘겨 재귀적으로 호출하는 것으로 repeat 을 완성시킨다.      
  - input 이 없는 경우 원래 Await 의 recv 를 실행시킨다. 재귀적 호출은 없다.
- 반복적인 Process 를 생성하는 기본적인 패턴2
  - 아래처럼 함수 자신이 재귀적인 go 함수를 가질 수도 있다.
```kotlin
fun sum(): Process<Double, Double> {
    fun go(acc: Double): Process<Double, Double> =
        Await { i: Option<Double> ->
            when (i) {
                is Some -> Emit(i.get + acc, go(i.get + acc))
                is None -> Halt<Double, Double>()
            }
        }
    return go(0.0)
}
```
```kotlin
>>> sum()(Stream.of(1.0, 2.0, 3.0, 4.0)).toList()
res4: chapter3.List<kotlin.Double> =    
 Cons(head=1.0, tail=Cons(head=3.0, tail=
 Cons(head=6.0, tail=Cons(head=10.0, tail=Nil))))
```
- ex 15.1) take, drop, takeWhile, dropWhile 을 구현하라
  - fun <I> take(n: Int): Process<I, I> = n 을 하나씩 줄이면서 재귀호출
  - fun <I> drop(n: Int): Process<I, I> = n 을 하나씩 줄이면서 drop 호 
  - fun <I> takeWhile(p: (I) -> Boolean): Process<I, I> = p 를 유지하며 재귀호출 stream 은 계속 소비됨
  - fun <I> dropWhile(p: (I) -> Boolean): Process<I, I> = p 를 유지하며 재귀호출 단 조건이 일치하지 않으면 Emit 을 생성하되 tail 에 다시 p 를 유지한 재귀호출을 함
```kotlin
take(2)(Stream.of(1.0, 2.0, 3.0, 4.0)).toList()
Cons(head=1.0, tail=Cons(head=2.0, tail=Nil))
```
- ex 15.2) count 를 구현하라
  - fun <I> count(): Process<I, Int> = sum 과 유사함
- ex 15.3) mean 을 구현하라
  - fun mean(): Process<Double, Double> = go가 sum 과 count 를 파라미터로 가지도록 한 뒤 재귀호출 함
- 위의 combinator 들을 패턴을 가지고 있다. 그래서 we can factor these patterns out into generic combinators.
  sum, count, mean 은 모두 같은 패턴을 공유하는데 이 패턴은 하나의 state 를 가지고, input 에 대한 응답으로 이 상태를 업데이트하는 state transition function 을 가지고, 하나의 아웃풋을 생산하는 패턴이다.  
  이를 loop 라는 컴비네이터로 정의할 수 있다. 
```kotlin
fun <S, I, O> loop(z: S, f: (I, S) -> Pair<O, S>): Process<I, O> =
    Await { i: Option<I> ->
        when (i) {
            is Some -> {
                val (o, s2) = f(i.get, z)
                Emit(o, loop(s2, f))
            }
            is None -> Halt<I, O>()
        }
    }
```
- ex 15.4) loop 를 사용해서 sum, count combinator 들을 구현하라.
  - fun sum(start: Double): Process<Double, Double> = loop(0.0, { i, s -> (i + s) to (i + s) })
  - fun <I> count(): Process<I, Int> = loop(0, { i, s -> (i + 1) to (i + 1) })
 
### 15.2.2 Combining multiple transducers by appending and composing
- This section deals with the composition of multiple processes.
- pipe: f pipe g As soon as values are emitted by f, they’re transformed by g.
- ex 15.5) pipe 를 구현하라.
  - infix fun <I, O, O2> Process<I, O>.pipe(g: Process<O, O2>): Process<I, O2> = 
    - pipe 는 각각의 단계에서 1:1 로 이어진다.
    - g 가 Halt 이면 Halt() 를 리턴한다.
    - g 가 Emit 이면 Emit(g.head, this pipe g.tail) 을 리턴한다. g 에 도달 해서야 Emit 으로 출력 될 수 있다.
    - g 가 Await 일때
      - f 가 Halt 이면 Halt<I, O>() pipe g.recv(None) 를 리턴한다. feed 할게 없음으로 g.recv(None) None 을 feed 한다. 한단계씩 진행한 state 를 다시 pipe 로 잇는다.     
      - f 가 Emit 이면 this.tail pipe g.recv(Some(this.head)) 를 리턴한다. g 에 f.head 를 feed 한다. 한단계씩 진행한 state 를 다시 pipe 로 잇는다.
      - f 가 Await 이면 Await { i -> this.recv(i) pipe g } 를 리턴한다. 새로운 Await 을 생성한다. 이 Await 은 나중에 f.recv 와 g 를 pipe 로 잇는 state 가 된다.  
- 어떤 function 이든 Process 안으로 태울 수 있는 lift 가 있음으로 map 을 구현할 수 있다.
```kotlin
fun <O2> map(f: (O) -> O2): Process<I, O2> = this pipe lift(f)
```
```kotlin
infix fun append(p2: Process<I, O>): Process<I, O> =
    when (this) {
        is Halt -> p2
        is Emit -> Emit(this.head, this.tail append p2)
        is Await -> Await { i: Option<I> ->
            (this.recv andThen { p1 -> p1 append p2 })(i) 
        // this.recv(i) 를 실행하면 p1: Process<I, O> 가 리턴되고 프로세스와 p2 를 append 한 process 를 리턴하는 Await 을 리턴한다.  
        }
    }
```
- append 를 사용해서 flatMap 을 구현할 수 있다. flatMap 에서 state 에 따라서 f 를 적용한다음 그것을 append 로 이어 붙인다.
```kotlin
fun <O2> flatMap(f: (O) -> Process<I, O2>): Process<I, O2> =
    when (this) {
        is Halt -> Halt()
        is Emit -> f(this.head) append this.tail.flatMap(f)
        is Await -> Await { i: Option<I> ->
            (this.recv andThen { p -> p.flatMap(f) })(i)
        }
      // this.recv(i) 를 실행하면 p: Process<I, O> 가 리턴되고 이 p에 대해서 다시 flatMap(f) 를 적용하는 Await 을 리턴한다.
    }
```
- pipe, append, flatMap 에서 this 가 Await 이면 Await 을 리턴한다. 
- Process 는 모나드이고 monad 인스턴스를 정의할 수 있다.
```kotlin
@extension
interface ProcessMonad<I, O> : Monad<ProcessPartialOf<I>> {
 
    override fun <A> unit(a: A): ProcessOf<I, A> = Emit(a)
 
    override fun <A, B> flatMap(
        fa: ProcessOf<I, A>,
        f: (A) -> ProcessOf<I, B>
    ): ProcessOf<I, B> =
        fa.fix().flatMap { a -> f(a).fix() }
 
    override fun <A, B> map(
        fa: ProcessOf<I, A>,
        f: (A) -> B
    ): ProcessOf<I, B> =
        fa.fix().map(f)
}
```
- 다음과 같이 type 파라미터를 설정해서 object 로 초기화 한다음 사용할 수 있다.
```kotlin
object pcm: ProcessMonad<Stream<Int>, Stream<Int>>

val prog = {
    val pc1 = pcm.unit(Stream.of(1))

}
```
- Process 는 input 을 받아서, grouping, mapping, filtering, folding 등으로 input 을 변환 시킬 수 있다.
  Process 는 거의 모든 Stream transformation 을 표현할 수 있으며, 그동안 input 이 어떻게 얻어지고 output 에 무슨일이 일어나는지에 대해서는 알지 못한다.
- ex) 15.6 mean 을 구현하라
  - fun mean(): Process<Double, Double> = zip(sum(), count()).map { (sm, cnt) -> sm / cnt }
  - fun <A, B, C> zip(p1: Process<A, B>, p2: Process<A, C>): Process<A, Pair<B, C>> = 두 프로세스의 Output 을 pair 로 묶는다. 
    p1 이 emit 일 경우 p2 또한 emit 이어야 한다. 같은 state 일 때만 zip 이 가능하다. 
  - fun <A, B> feed(oa: Option<A>, p1: Process<A, B>): Process<A, B> = p1 의 state 에 따라 oa 를 feed 한다. 
- ex) 15.7 zipWithIndex 를 구현하라
  - fun <I, O> Process<I, O>.zipWithIndex(): Process<I, Pair<Int, O>> = zip(count<I>().map { it - 1 }, this)
- ex) 15.8 여러 타입의 exists 를 구현하라
  - 하나만 찾으면 그 결과를 리턴하는것 Stream(true) -> trim 으로 구현 가능한가? 
  - true 가 리턴 될 때 까지 각 아이템에 대한 결과를 리턴하고 종료 Stream(false, false, false, true)
  - true 가 리턴 될 때 까지 각 아이템에 대한 결과를 리턴하고 끝나지않고 마지막 까지 true 가 계속되는 것 Stream(false, false, false, true, true)
  - fun <I> exists(f: (I) -> Boolean): Process<I, Boolean> = exists 는 Await 이다. input 을 받으면 f 로 검사한 후 매치되면 tail 에 exist 를 재귀 호출하는 Emit 을 리턴한다.
  - fun <I> existsAndHalt(f: (I) -> Boolean): Process<I, Boolean> = 일치하면 tail 이 Halt() 인 Emit 을 리턴한다. 일치하지 않으면 existsAndHalt 를 재귀호출 한다. 
```kotlin
val stream = Stream.of(1, 3, 5, 6, 7)
val p = existsAndHalt<Int> { i -> i % 2 == 0 }
val result = p pipe takeWhile { it } (stream)
[true]
```
- `count() pipe exists { it > 40000 }` 4만 라인이 넘는지 확인하는 프로그램은 이처럼 간단해 지고 filter 를 붙이거나 다른 변환도 이 pipeline 에 쉽게 붙일 수 있다.  

### 15.2.3 Stream transducers for file processing
- 파일의 각 line 을 process 에 feed 한다. Emit 일 때 fn 을 실행해서 새로운 acc 값을 만든다.  Process 가 Halt 일때 비로서 acc 값을 리턴한다.
- processFile 은 file 을 사용해서 Process 를 drive 할 수 있다. 이 드라이버는 결과물로 stream 을 생성하는 대신에 Process 가 emit 하는 내용을 누적시킨다. 
  이 누적은 List 의 foldLeft 와 유사하다. processFile 의 결과는 IO<B> 이다. 
```kotlin
fun <A, B> processFile(
    file: File,
    proc: Process<String, A>,
    z: B,
    fn: (B, A) -> B
): IO<B> = IO {
 
    tailrec fun go(
        ss: Iterator<String>,
        curr: Process<String, A>,
        acc: B
    ): B =
        when (curr) {
            is Halt -> acc
            is Await -> {
                val next =
                    if (ss.hasNext()) curr.recv(Some(ss.next()))
                    else curr.recv(None)
                go(ss, next, acc)
            }
            is Emit -> go(ss, curr.tail, fn(acc, curr.head))
        }
 
    file.bufferedReader().use { reader ->
        go(reader.lines().iterator(), proc, z)
    }
}
```
```kotlin
val proc = count<String>() pipe exists { it > 40000 }
processFile(f, proc, false) { a, b -> a || b } // 결과는 계속해서 누산 (accumulate) 된다. 
```
- 15.9) 파일의 Fahrenheit 를 Celsius 로 바꾸서 새로운 파일에 저장하는 function 을 구현하라.
  - fun convert(infile: File, outfile: File): File = outfile.bufferedWriter().use(block: (BufferedWriter) -> File) 을 사용해서
    block 에서 bw 를 사용할 수 있도록 한다. 이 block 에서 누산기 fn 은 bw를 사용해서 outputFile 에 결과를 쓰고 file 을 돌려준다.
    processFile 의 process 는 lift 로 생성한 Fahrenheit 를 Celsius 로 바꾸는 반복되는 Process 이다.
    processFile 의 초기 값은 outputFile 자신이다.
    따라서 processFile 를 실행하면 인수 inputFile 을 한 줄씩 읽어서 인수 process 에 의해 변환된 결과가 outputFile 에 쓰이게 된다. 

## 15.3 An extensible process type for protocol parameterization
- 이전 섹션에서 스트림의 값을 품는 환경과 건택스트를 암시적으로 가정하는 제한된 프로세스 타입을 정의했다. 이 타입의 단점은 드라이버와 통신할때 
  Halt, Emit, Await 의 3 가지 instructions 를 가지는 고정된 프로토콜을 사용한다고 가정하는 것이다. 
- 이 섹션에서는 드라이버의 요청을 발행하는데 사용되는 프로토콜을 parameterizing 하는 것이다.
```kotlin
@higherkind
sealed class Process<F, O> : ProcessOf<F, O> {
    companion object {
        data class Await<F, A, O>(
            val req: Kind<F, A>, // Await 은 이제 A 가 아니라 Kind 를 다룬다. req 를 한 번 더 wrapping 할 수 있다. 
            val recv: (Either<Throwable, A>) -> Process<F, O> // recv 는 A 만 받는게 아니라 error 를 핸들링 하기 위해 Either<Throwable, A> 를 받는다.
        ) : Process<F, A>()
 
        data class Emit<F, O>(
            val head: O,
            val tail: Process<F, O>
        ) : Process<F, O>()
 
        data class Halt<F, O>(val err: Throwable) : Process<F, O>() // process 가 error 를 핸들링 하는 능력이 생겼음으로 error 또는 정상적인 종료가 될 수 있다.
 
        object End : Exception() // 정상 종료를 나타내는 오브젝트 Exception 을 확장한 이유는 kotlin 의 exception 메카니즘을 사용하기 위함.
 
        object Kill : Exception() // 강제적인 종료를 나타내는 오브젝트
    }
 
}
```
- Free<F, A> 와 다른점은 Process<F, O> 에서 O 는 Stream<O> 라는 것이다.
  에 F는 Free<F, A> 에서 Suspend 에서 사용되는 것과 같이 Process<F, O> 에서 Await 에서 사용되며 같은 롤을 같는다. (instruction sets 을 표현함)
  Process 는 Emit 을 여러번 요청할 수 있지만, Free 는 항상 그것의 마지막 Return 안에 하나의 답을 가진다.
  Free 는 Return 과 함께 종료되지만 Process 는 Halt 와 함께 종료된다.
- driver 에 따라서 protocol 이 fix 되었는데, 이 새로운 Process 타입에서는 request 에 사용되는 프로토콜을 파라미터화 해서 더 일반적인 타입이 되었다.
  이제 Await 은 req 를 가진다. 이전에 driver 에 의해서 제공되는 것이 파라미터화 되었다.
- Process 타입을 더 일반적인 타입으로 재 정의 했기 때문에, 이전의 프로세스 타입을, Process 의 하나의 인스턴스로 “single-input Process” Process1 이라고 부를 수 있다.
- 새로운 Process 타입 또한 F 에 관계없이 append, map, filter 를 정의할수 있다. append 는 더 일반적인 onHalt 라는 함수로 정의할 수 있다.  
```kotlin
fun onHalt(f: (Throwable) -> Process<F, O>): Process<F, O> =
    when (this) {
        is Halt -> tryP { f(this.err) }
        is Emit -> Emit(this.head, tail.onHalt(f))
        is Await<*, *, *> ->
            awaitAndThen(req, recv) { p: Process<F, O> ->
                p.onHalt(f)
            }
    }
 
fun append(p: () -> Process<F, O>): Process<F, O> =
    this.onHalt { ex: Throwable ->
        when (ex) {
            is End -> p()
            else -> Halt(ex)
        }
    }.fix()
```
- onHalt 에서는 this 가 halt 일때만 ex 를 자신의 argument f 에 feed 한다. 따라서 이를 이용해서 우리는 더 나아간 로직(further logic)과 함께 process 를 확장할 수 있고,
  process 의 termination 이유에 접근 할 수 있다. 
- tryP 는 Process 의 evaluation 을 품고 어떤 exception 이든지 발생하면 그것을 Halt 로 되돌려 준다. 
  onHalt 의 인수 f 가 Process 를 돌려주는 함수이기는 하나 (further logic) 이기 때문에 exception 이 발생하면 catch 해서 Halt(es) 로 되돌려줘야 한다. 
```kotlin
fun <F, O> tryP(p: () -> Process<F, O>): Process<F, O> =
    try {
        p()
    } catch (e: Throwable) {
        Halt(e)
    }
```
- onHalt 를 사용해서 exception 을 다루는 것은 resource safety 를 위한 필수적인 것이다. 우리의 목표는 우리의 라이브러리 사용자에게 exception 을 넘기기 보다는 
  모든 exception 을 캐치하고 다루는 것이다. exception 을 발생시키는 combinator 는 몇가지 없으며 이것들을 안전하게 다룬다면, Process 를 사용하는 모든 프로그램이
  resource safety 하다는 것을 보증할 수 있다.
- awaitAndThen 은 두가지 목적을 가지고 있다. 하나는 런타입에서의 타입 소거로 인해 손실된 타입정보를 다시 도입하는 것이며,
  나머지 하나는 recv function 에 fn 을 이어 붙여서 새로 생성하는 Await 의 recv 로 설정하는 것이다.
- awaitAndThen 은 새로운 Await 을 생성한다!
```kotlin
fun <F, A, O> awaitAndThen(
    req: Kind<Any?, Any?>,
    recv: (Either<Throwable, Nothing>) -> Process<out Any?, out Any?>,
    fn: (Process<F, A>) -> Process<F, O>
): Process<F, O> =
    Await(
        req as Kind<F, Nothing>, // 인수로 들어온 req, recv 는 타입이 없지만 awaitAndThen 에 <F, A, O> 가 정의되어있고 as 를 사용해 타입정보를 도입한다. 도입된 타입정보를 기반으로 Await 을 생성한다.
        recv as (Either<Throwable, A>) -> Process<F, A> andThen fn // 기존 recv 의 실행 결과를 fn 에 넣는 함수를 생성 해서 새로운 Await 의 인수로 넘긴다.
    ).fix()
```
- flatMap 에서 f 가 exception 을 뱉는지 알 수 없음으로 이또한 tryP 로 감싸야 한다.
```kotlin
fun <O2> flatMap(f: (O) -> Process<F, O2>): Process<F, O2> =
  when (this) {
    is Halt -> Halt(err)
    is Emit -> tryP { f(head) }.append { tail.flatMap(f) }
    is Await<*, *, *> ->
      awaitAndThen(req, recv) { p: Process<F, O> ->
        p.flatMap(f)
      }
  } 
```
- map 또한 비슷한 패턴을 따른다.
```kotlin
fun <O2> map(f: (O) -> O2): Process<F, O2> =
    when (this) {
        is Halt -> Halt(err)
        is Emit -> tryP { Emit(f(head), tail.map(f)) }
        is Await<*, *, *> ->
            awaitAndThen(req, recv) { p: Process<F, O> ->
                p.map(f)
            }
    }
```
### 15.3.1 Sources of stream emission
- 15.2 에서는 파일을 읽으면서 process 를 앞으로 drive 하기 위한 분리된 function 을 직접 작성해야 했지만 새로운 Process 에서는 타입 파라미터가 
  추가됨으로 Process<ForIO, O> 를 사용해서 source 를 표현할 수 있게 되었다. ( Process<I, O> 와 Process<ForIO, O> 는 다르다. ForIO 는 type constructor 다. )
- Process<ForIO, O> 가 어떻게 O 를 생성하는 Source 가 되는지 살펴보기 위해서 Await constructor 에서 Kind<F,A> 를 IO<A> 로 대체해 보자.
```kotlin
data class Await<ForIO, A, O>(
    val req: IO<A>,
    val recv: (Either<Throwable, A>) -> Process<ForIO, O>
) : Process<ForIO, O>()

```
- 외부세계의 어떤한 requests 든 IO 의 flatMaping 을 하거나 unsafePerformIO 을 실행함으로써 얻어 질수 있다. 
  이를 사용해서 recv 실행할 수 있다. 아래는 runLog driver 이다. 이전 색션의 processFile driver 와 다른점은 process 자신이 source 도 표현한 다는것이다.
  processFile 가 accumulator 를 설정 할 수 있다는 부분에서는 조금 더 유연하다.
  driver 가 tray finally 구문을 사용해서 E 는 마지막에 반드시 shutdown 된다. (IO 가 다루는 resource 의 closing 은 runLog 의 책임이 아니다.) 
```kotlin
fun <O> runLog(src: Process<ForIO, O>): IO<Sequence<O>> = IO {
 
    val E = java.util.concurrent.Executors.newFixedThreadPool(4)
 
    tailrec fun go(cur: Process<ForIO, O>, acc: Sequence<O>): Sequence<O> =
        when (cur) {
            is Emit ->
                go(cur.tail, acc + cur.head)
            is Halt ->
                when (val e = cur.err) {
                    is End -> acc
                    else -> throw e
                }
            is Await<*, *, *> -> {
                val re = cur.req as IO<O>  
                val rcv = cur.recv
                    as (Either<Throwable, O>) -> Process<ForIO, O>
                val next: Process<ForIO, O> = try {
                    rcv(Right(unsafePerformIO(re, E)).fix())
                } catch (err: Throwable) {
                    rcv(Left(err))
                }
                go(next, acc)
            }
        }
    try {
        go(src, emptySequence())
    } finally {
        E.shutdown()
    }
}
```
- file 을 다루는 process 는 다음과 같은 프로그램이 된다.
```kotlin
val p: Process<ForIO, String> =
    await<ForIO, BufferedReader, String>(
        IO { BufferedReader(FileReader(fileName)) }
    ) { ei1: Either<Throwable, BufferedReader> ->
        when (ei1) {
            is Right -> processNext(ei1)
            is Left -> Halt(ei1.value)
        }
    }
 
private fun processNext(
    ei1: Right<BufferedReader>
): Process<ForIO, String> =
    await<ForIO, BufferedReader, String>(
        IO { ei1.value.readLine() }   // req 도 계속 계산해야 한다.
    ) { ei2: Either<Throwable, String?> ->
        when (ei2) {
            is Right ->
                if (ei2.value == null) Halt(End)
                else Emit(ei2.value, processNext(ei1))
            is Left -> {
                await<ForIO, Nothing, Nothing>(
                    IO { ei1.value.close() }
                ) { Halt(ei2.value) }
            }
        }
    }
```
- 15.10) 좀더 일반적인 runLog 를 구현하라.
  protocol 은 monad 여야 한다.
  fun <F, O> Process<F, O>.runLog(MC: MonadCatch<F>): Kind<F, Sequence<O>> = 
    기존 runLog 와 다르게 return type 이 Sequence<O> 가 아니라, Kind<F, Sequence<O>> 이다.
    evaluation 은 MC.flatMap 을 통해서 연결되고 (not stack-safe) , MC.attempt 를 통해 req 에서 값을 얻어낸다.
```
fun <F, O> Process<F, O>.runLog(
    MC: MonadCatch<F>
): Kind<F, Sequence<O>> =
 
    SOLUTION_HERE()
 
interface MonadCatch<F> : Monad<F> {
    fun <A> attempt(a: Kind<F, A>): Kind<F, Either<Throwable, A>>
    fun <A> fail(t: Throwable): Kind<F, A>
}
```
### 15.3.2 Ensuring resource safety in stream transducers (transducers 에서 resource safety 보장하가.)
- lines: Process <ForIO, String> representing the lines of some large file. This process is a source or producer, and it implicitly references a file handle resource.
- Ensuring resource safety 를 위해서는 다음과 같은 두가지 룰이 필요하다.
  - A producer should free any underlying resources as soon as it knows it has no further values to produce, whether due to normal exhaustion or an exception.
  - Any process d that consumes values from another process p must ensure that cleanup actions of p are performed before d halts.
- 두번째 driver 는 두번째 룰에 대한 책임을 질 수 없다. 왜냐하면 process 가 어떻게 연결되고 언제 끝나는지 알 수 없기 때문이다. 
- 프로세스 p 는 다음 세가지 이유로 종료될 수 있다. 어떤 이유든 resource 는 close 되어야 한다. 
  - 생산자 고갈, 기저 소스가 더 이상 값을 내보낼 수 없음을 나타내는 'End'로 신호를 보냄
  - 강제 종료, p의 소비자가 소비를 마쳤음을 나타내기 위해 'Kill'로 신호를 보냄, 이는 아마도 생산자 p가 고갈되기 전에 발생할 수 있음
  - 비정상 종료, 생산자나 소비자 중 어느 한쪽에서 발생하는 어떤 'e: Throwable' 때문에 발생
- 이전 process 가 어떻게 종료되었는지 상관없이 process 를 append 할 수 있는 onComplete 컴비네이터는 다음과 같다. 
```kotlin
fun onComplete(p: () -> Process<F, O>): Process<F, O> =
    this.onHalt { e: Throwable ->
        when (e) {
            is End -> p().asFinalizer()
            else -> p().asFinalizer().append { Halt(e) }
        }
    }.fix()
```
- 인수 p 는 this 가 halt 일때 항상 실행된다. 만약 this 가 error 를 가지고 있으면 이 컴비네이터에서 p 를 실행한 후에 append 를 통해 다시 발생시킨다. 

```kotlin
private fun asFinalizer(): Process<F, O> =
    when (this) {
        is Emit -> Emit(this.head, this.tail.asFinalizer())
        is Halt -> Halt(this.err)
        is Await<*, *, *> -> {
            await<F, O, O>(this.req) { ei ->
                when (ei) {
                    is Left ->
                        when (val e = ei.value) {
                            is Kill -> this.asFinalizer()
                            else -> this.recv(Left(e))
                        }
                    else -> this.recv(ei)
                }
            }
        }
    }
```
- onHalt 자체가 halt 는 process 자체를 변경하는 컴비네이터 이다. this 가 halt 일때 f 를 사용해서 자신을 변경한다. 자신이 emit 이거나 await 일때는 f 의 적용을 보류했다가 나중에 적용한다. 
  asFinalizer 도 마찬가지로 process 자체를 변경하는 컴비네이터 이다. this 가 await 이고 kill 을 인수로 받았을때 자기 자신을 실행하는 프로세스로 만든다. 
  즉 `p1.onComplete(p2)` 에서 stream 의 consumer 가 일찍 종료되기를 희망하는 경우에도 p2 는 항상 실행된다. 
- 이 모든 것을 조합하면 resource 컴비네이터를 정의할 수 있다.
```kotlin
fun <R, O> resource(
    acquire: IO<R>,
    use: (R) -> Process<ForIO, O>,
    release: (R) -> Process<ForIO, O>
): Process<ForIO, O> =
    eval(acquire)
        .flatMap { use(it).onComplete { release(it) } }
```
- eval 은 `await<ForIO, BufferedReader, String>(IO { BufferedReader(FileReader(fileName)) }) { ..}` 와 같은 역할을 한다.
  flatMap 으로 process 의 O 을 읽어들인다. use 를 통해 O 를 사용하고, 마지막에 O 를 release 한다.
- ex 15.2) implement eval and evalDrain
```kotlin
fun <F, A> eval(fa: Kind<F, A>): Process<F, A> =
  await<F, A, A>(fa) { ea: Either<Throwable, Nothing> ->
    when (ea) {
      is Right<A> -> Emit(ea.value, Halt(End))
      is Left -> Halt(ea.value)
    }
  }

fun <F, A, B> evalDrain(fa: Kind<F, A>): Process<F, B> =
  eval(fa).drain()

fun <F, A, B> Process<F, A>.drain(): Process<F, B> =
  when (this) {
    is Halt -> Halt(this.err)
    is Emit -> this.tail.drain()
    is Await<*, *, *> ->
      awaitAndThen<F, A, B>(
        this.req,
        { ei: Either<Throwable, Nothing> -> this.recv(ei) },
        { it.drain() }
      )
  }
```
- runLog 에서의 process 와 다른 점은 그 process 는 처음 req 를 셋팅한 다음 계속된 처리까지 process 로 나타냈지만,
  eval 의 경우 값이 있으면, Emit 타입의 프로세스를 하나 리턴함으로 flatMap 으로 프로세스를 연결해야 한다. 
- drain 의 경우 Process 에서 발생하는 Emit 들을 흡수하고, Process 의 마지막 Halt 만 리턴한다. Process 가 Await 을 가지고 있을 경우 그것을 실행한 후에 실행결과 Process 에 다시 drain 을 적용한다.   
- evalDrain 은 fa 에서 값을 읽어드린 후 발생하는 Emit 은 무시하고 마지막 Halt 만 리턴한다. 
- 이를 이용하여 다음과 같은 lines 함수를 정의 할 수 있다. 
```kotlin
fun lines(fileName: String): Process<ForIO, String> =
    resource(
        IO { BufferedReader(FileReader(fileName)) },
        { br: BufferedReader ->
 
            val iter = br.lines().iterator()
 
            fun step() = if (iter.hasNext()) Some(iter.next()) else None // step 을 실행하면 iter.next() 에 의해 다음 라인으로 변경됨
 
            fun lns(): Process<ForIO, String> {
                return eval(IO { step() }).flatMap { ln: Option<String> -> // eval 과 flatMap 사용해서 외부세계의 값을 process 로 불러드릴 수 있다.  
                    when (ln) {
                        is Some -> Emit(ln.get, lns()) // tail 에 lns 를 이어 붙이도록 함. 
                        is None -> Halt<ForIO, String>(End)
                    }
                }
            }
 
            lns()
        },
        { br: BufferedReader -> evalDrain(IO { br.close() }) } 
        // release 는 br 을 Process 로 승격 시킨 후 emit 을 무시하고 process 의 err 만을 계승한 Halt 프로세스를 돌려주는 함수다. 
    )
```
- use 내부에서 프로세스에서 사용할 값을 생겅하는 함수를 정의한 다음, 
  eval 과 flatMap 을 사용해서 외부세계의 값을 프로세스 내부로 불러들여 처리하는 함수를 정의하고 이 안에서 다시 이 함수를 재귀호출 함으로써 일련의 프로세스를 정의할 수 있다.
- resource 함수를 사용해서 resource 를 안전하게 정리할 수 있다. 이제 남은 것은 새로운 프로세스 타입에 대하여 pipe 나 다른 컨슈머가 소비가 끝났을때 잘 종료되도록 하는것이다.

### 15.3.3 Applying transducers to a single-input stream (하나의 input stream 에 transducers 적용하기)
- the process Process1 always assumes the environment or context of a single stream of values that allows us to apply such transformations.
- 'class Is<I> : IsOf<I>' 이 쐐기를 사용해 Process 를 사용해서 Process1 을 표현할 수 있다.
  'typealias Process1<I, O> = Process<ForIs, O>'
- Process1 을 생성하는 await1 은 새로운 Process 타입이 error 를 핸들링 하는 능력이 생겼음으로 fallback 을 인수로 전달할 수 있다. 이것의 default 값은 halt1<ForIs, O>() 이다. 
```kotlin
fun <F, A, O> await(
            req: Kind<Any?, Any?>,
            recv: (Either<Throwable, Nothing>) -> Process<out Any?, out Any?>
        ): Process<F, O> = Await(
            req as Kind<F, A>,
            recv as (Either<Throwable, A>) -> Process<F, O>
        )
```
```kotlin
fun <I, O> await1(
  recv: (I) -> Process1<ForIs, O>,
  fallback: Process1<ForIs, O> = halt1<ForIs, O>()
): Process1<I, O> =
  Await(Is<I>()) { ei: Either<Throwable, I> -> // Propagating Is() as req forces I in recv.
    when (ei) {
      is Left ->
        when (val err = ei.value) {
          is End -> fallback
          else -> Halt(err)
        }
      is Right -> Try { recv(ei.value) }
    }
  }

fun <I, O> halt1(): Process1<ForIs, O> =
  Halt<ForIs, O>(End).fix1()

fun <I, O> emit1(
  head: O,
  tail: Process1<ForIs, O> = halt1<ForIs, O>()
): Process1<ForIs, O> =
  Emit<ForIs, O>(
    head,
    tail.fix1()
  ).fix1()
```
- Process1 을 인수로 받는 pip 도 구현가능하다. 단 오른쪽 프로세스가 종료되기전에 왼쪽 프로세스가 cleanup 액션을 수행하도록 해야 한다.
```kotlin
infix fun <O2> pipe(p2: Process1<O, O2>): Process<F, O2> =
    when (p2) {
        is Halt -> 
            this.kill<O2>()
                .onHalt { e2 ->
                    Halt<F, O2>(p2.err).append { Halt(e2) } 
                // Halt<F, O2>(p2.err).append { Halt(e2) } 는 아래 함수와 같다. 
                // p2.err 가 End 면, this 의 e2 를 인수로 가진 Halt 를 돌려준다.
                // p2.err 가 End 가 아니면, 그대로 p2.err 를 되돌려 준다.  
                  /*
                  Try {
                    { e ->
                         when (e) {
                          is End -> Try(Halt(e2))
                          else -> Halt(e)
                         }
                     }(p2.err)
                   }
                  * */
                }
        is Emit ->
            Emit(p2.head, this.pipe(p2.tail.fix1()))
        is Await<*, *, *> -> {
            val rcv =
                p2.recv as (Either<Throwable, O>) -> Process<F, O2>
            when (this) {
                is Halt ->
                    Halt<F, O2>(this.err) pipe
                        rcv(Left(this.err)).fix1()
                is Emit ->
                    tail.pipe(Try { rcv(Right(head).fix()) }.fix1())
                is Await<*, *, *> ->
                    awaitAndThen<F, O, O2>(req, recv) { it pipe p2 }
            }
        }
    }
```
```kotlin
fun <O2> kill(): Process<F, O2> =
    when (this) {
        is Await<*, *, *> -> {
            val rcv =
                this.recv as (Either<Throwable, O>) -> Process<F, O2>
            rcv(Left(Kill)).drain<O2>()
                .onHalt { e ->
                    when (e) {
                        is Kill -> Halt(End)
                        else -> Halt(e)
                    }
                }
        }
        is Halt -> Halt(this.err)
        is Emit -> tail.kill()
    }
 
fun <O2> drain(): Process<F, O2> =
    when (this) {
        is Halt -> Halt(this.err)
        is Emit -> tail.drain()
        is Await<*, *, *> ->
            awaitAndThen<F, O2, O2>(req, recv) { it.drain() }
    }
```
- Emit 만이 tail 을 갖는다. Await 은 중첩된 expression 이다.
- Process.filter() 함수는 Process1 을 되돌려준다. pipe 는 Process1 과 연결할 수 있음으로 Process 의 filter 는 다음과 같이 표현할 수 있다.
```kotlin
fun filter(f: (O) -> Boolean): Process<F, O> =
    this pipe Process.filter(f)
```

### 15.3.4 Multiple input streams
- Tee, which combines two input streams in some way, can also be expressed as a Process
- Process<F, O> 는 이미 충분히 유연하다. Is 타입을 정의한다음 F 에 대입함으로 Process1 을 정의 했듯, T 타입을 정의한다음 F 에 대입한으로 Tee 를 정의할 수 있다.
  - type constructor F 는 req 를 표현함으로 두개의 stream 을 합치는 input 을 표현하기 위해서는 그것을 표현하기 위한 T 타입의 정의가 필요하다. 
```kotlin
@higherkind
sealed class T<I1, I2, X> : TOf<I1, I2, X> {
 
    companion object {
        fun <I1, I2> left() = L<I1, I2>()
        fun <I1, I2> right() = R<I1, I2>()
    }
 
    abstract fun get(): Either<(I1) -> X, (I2) -> X>
 
    class L<I1, I2> : T<I1, I2, I1>() {
        override fun get(): Either<(I1) -> I1, (I2) -> I1> =
            Left { l: I1 -> l }
    }
 
    class R<I1, I2> : T<I1, I2, I2>() {
        override fun get(): Either<(I1) -> I2, (I2) -> I2> =
            Right { r: I2 -> r }.fix()
    }
}
```
- `typealias Tee<I1, I2, O> = Process<ForT, O>`
- awaitL, awaitR, emitT, haltT 컴비네이터 를 정의해서 Tee 타입의 프로세스를 쉽게 생성할 수 있다.
- zipWith 컴비네이터를 정의 해서 f:(I1, I2) -> O 를 태운 Tee 타입의 프로세스를 쉽게 생성할 수 있다. 
  - Tee Process 를 만든 다음 여기에 다른 프로세스를 붙임으로써 프로그램을 만들 수 있다.
- zip 컴비네이터를 정의해서 두 input 을 하나로 묶어 pair 로 만든느 Tee 타입의 프로세스를 쉽게 생성할 수 있다.
- fun <F, I1, I2, O> tee(p1: Process<F, I1>, p2: Process<F, I2>, t: Tee<I1, I2, O>): Process<F, O> 
  tee 컴비네이터를 정의 해서 두 프로세스를 하나의 프로세스로 합칠 수 있다. 이는 pipe 와 유사하게 동작한다.
  다른점은 tee 에서는 req 가 left 인지 right 인지 확인한 다음 그에 따라 동작한다는 것이다.
  t 의 req 가 Left 이면, p1 을 소비한다. Right 이면 p2 를 소비한다.  

### 15.3.5 Sinks for output processing
- `typealias Sink<F, O> = Process<F, (O) -> Process<F, Unit>>` A Sink<F, O> provides a sequence of functions to call with the input type O.
```kotlin
fun fileW(file: String, append: Boolean = false): Sink<ForIO, String> =
    resource(
        acquire = IO { FileWriter(file, append) },
        use = { fw: FileWriter ->
            constant { s: String ->  // s 을 인수로 받는 가지는 함수, eval 을 사용해서 파일에 s 를 쓰는 IO 를 Process<IO, Unit> 으로 승격시킨다.   
                eval(IO {
                    fw.write(s)
                    fw.flush()
                })
            }
        },
        release = { fw: FileWriter ->
            evalDrain(IO { fw.close() })
        }
    )//fun <F, A> eval(fa: Kind<F, A>): Process<F, A> =
// 값 a 를 계속 해서 출력하는 스트림을 만드는 함수, 여기에 String -> Process<IO, Unit> 가 인수로 적용된다.
// 따라서 String -> Process<IO, Unit> 를 f 라 치면 constant(f) 는 eval(IO { f }.flatMap { Emit(f, constant(f)) } 가 되고
// constant(f) 리턴 타입은 Process<ForIO, String -> Process<IO, Unit>> 이는 Sink<ForIO, String> 과 같다.
fun <A> constant(a: A): Process<ForIO, A> =
    eval(IO { a }).flatMap { Emit(it, constant(it)) }
```
- 이렇게 만들어진 Sink 는 파일에 s 를 쓰는 함수 수식을 품고 있는 Emit 의 stream 이다. 쓰기의 처리과정에 대한 표현식(expression) 으로
  transducer 를 사용해 다른 Process (읽기의 처리과장에 대한 표현식) 함께 묶고 실행 함으로써 실제 스기 처리과정이 완료될 수 있다.
```kotlin
// Tee 의 constructor 인 zipWith 과 다르다. p1 과 p2 그리고, 나중에 Tee 에 태울 f 를 사용해 두 프로세스를 하나의 프로세스로 만든다.  
fun <F, I1, I2, O> zipWith( 
    p1: Process<F, I1>,
    p2: Process<F, I2>,
    f: (I1, I2) -> O
): Process<F, O> =
    tee(p1, p2, zipWith(f)) // 두 프로세스를 하나로 묶는 tee 를 사용한다. 여기서 Tee 의 constructor 인 zipWith 를 사용한다. 

 
fun <F, O> Process<F, O>.to(sink: Sink<F, O>): Process<F, Unit> =
    join(
        zipWith(this, sink) { o: O, fn: (O) -> Process<F, Unit> ->
            fn(o)
        } 
      // zipWith 를 사용해 this 와 sink 를 하나의 프로세스로 묶는다. Sink<F, O> 는 Process<F, (O) -> Process<F, Unit>> 이고
      // 나중에 Tee 에 태울 f 를 `{ o: O, fn: (O) -> Process<F, Unit> -> fn(o) }` 로 설정한다.  
)
```
- 15.12) `fun <F, O> join(p: Process<F, Process<F, O>>): Process<F, O>` 을 구현하라.
  - p.flatMap{it} // 중첩된 것을 하나로 합친다.
```kotlin
fun converter(inputFile: String, outputFile: String): Process<ForIO, Unit> =
    lines(inputFile)
        .filter { !it.startsWith("#") }
        .map { line -> fahrenheitToCelsius(line.toDouble()).toString() }
        .pipe(intersperse("\n"))
        .to(fileW(outputFile))
        .drain()
```

### 15.3.6 Hiding effects in effectful channels
- Channel is useful when a pure pipeline must execute an I/O action as one of its stages.
- `typealias Sink<F, O> = Process<F, (O) -> Process<F, Unit>>`
- `typealias Channel<F, I, O> = Process<F, (I) -> Process<F, O>>`
```kotlin
fun query(
    conn: IO<Connection>
): Channel<ForIO, (Connection) -> PreparedStatement, Map<String, Any>>
```
- query 는 lines 처럼 구현되어야 한다.
- The Channel gives us an elegant solution for encapsulating effects such as I/O processing 
  without leaking its inherent concerns beyond the process’s boundaries.
  - IO 작업을 Channel 안으로 encapsulating 시킨다. 클라이언트가 이것들을 다루게 하지 않는다.

### 15.3.7 Dynamic resource allocation
- Process 를 사용하면, 안전하게 동적으로 자원을 사용할 수 있다. 동적으로 자원을 할당한다는 것은 필요할때 필요한 만큼만 사용한다는 것을 의미한다. 
- lines 는 안전하게 file 읽어서 각 라인은 input stream 으로 만드는 컴비네이터 이다. driver 는 순수하게 계산만을 다룬다.
- it is now straightforward to wire together such imperative-style code.

### 15.4 Application of stream transducers in the real world
- 수많은 프로그램을 stream processing 으로 표현할 수 있다.
- We can model a stream transducer as a state machine to await, emit, and halt at discreet points when consuming and transforming a stream of data.



## 요약
IO 블록 안에서 List 와 같이 연속된 요소를 추상화한 모델을 사용해서 composable 한 스타일의 프로그램을 작성하고 싶다.
( monolithic loop 에서는 concern 이 뒤섞인다.)
왜냐하면 더 유연한게 변경할 수 있기 때문이다. kotlin 에서 제공하는 Sequence 를 사용해 봤으나, Sequence 를 리턴하는 IO 는
pure value 가 아니다. IO 의 invoke 를 실행한 후에도 Sequence 는 확정되지 않는다. 따라서 Process 라는 타입을 정의해서 IO 타입때와 마찬가지로
List 와 같이 연속된 요소를 추상화한 모델을 사용할때 연산을 컴파일러에게 맞기지 않고 직접 컨트롤 할 수 있도록 했다.

Process 의 driver (invoke) 를 어떻게 구현하느냐에 따라서 Stream 뿐만 아니라 List 와 같이 연속된 요소를 추상화한 모델 무엇이든 Process 와 함께 처리할 수 있다.
Process 의 컴비네이터들을 정의했다. lift, pipe 와 append 를 정의하고 다시 이를 사용해 map 과 flatMap 을 정의했다. Process 는 모나드이다.
file 을 다룰 수 있는 Process 의 driver 인 processFile 을 정의했다. 이를 사용하면 파일에 각 라인에 process 를 적용하고 누산한 결과를 얻을 수 있다.
processFile 은 새로운 Process 타입을 정의하기 전에 driver 에서 직접 source 를 다룬다. 새로운 Process 타입에서는 source 가 일반화 되어 있다. 

type parameter 를 가지는 Process 타입을 정의 driver 에 따라서 protocol 이 fix 되었는데 새로운 타입에서는 await 이 req: Kind<F,A> 를 가짐으로써
input protocol 이 일반화 된 Process 타입이 되었다. 이전의 Process 타입은 일반화된 Process 타입의 한가지 인스턴스로써 하나의 input 을 가지는 Process 라는 의미로
Process1 이라고 부르게 되었다. 새 Process 타입은 또한 error handling 기능도 가진다. onHalt 라는 컴비네이터를 구현함으로써 further logic 을 추가할 수 있게되었고,
기존 append 를 포함해 여러  combinator 들 또한 onHalt 를 통해서 다시 구현할 수 있다.
onHalt 자체가 halt 는 process 자체를 변경하는 컴비네이터 이다. this 가 halt 일때 f 를 사용해서 자신을 변경한다. 자신이 emit 이거나 await 일때는 f 의 적용을 보류했다가 나중에 적용한다.

- 15.3.1 Sources for stream emission
  - Process<ForIO, O> 를 사용하면 IO<A> 에서 A 를 얻을 수 있다.
    input 를 IO 타입으로 감싼 프로세스를 Sequence 로 출력하는  runLog 라는 driver 를 구현했다. 이제 source 는 프로세스에서 표현횐다.
    결과 적으로 IO 내에서 Process 를 사용하는게 아니라 Process 가 input 으로써 IO 를 품는 형태가 되었다.
    더 일반화된 runLog 라는 driver 를 구현했다. input 의 wrapper 가 monad 라면, runLog 를 통해 계산을 할 수 있다.
- 15.3.2 Ensuring resource safety in stream transducers
  - Process<ForIO, String> 를 생성하는 lines 라는 함수 있다. 이 함수는 resource-safe 하게 파일의 각 라인을 Emit 하는 Process 들을 생성한다.
- 15.3.3 Applying transducers to a single-input stream 
  - Process 한단계 추상화된 타입니다. Is 타입을 사용하면 single-input stream 을 다루는 Process1 을 정의할 수 있다. 기존에 정의한 transducers 쉽게 다시 정의할 수 있다.
- 15.3.4 Multiple input streams 
  - Tee 는 두개의 프로세스를 하나로 결합하는 타입으로, Is 타입을 정의한다음 F 에 대입함으로 Process1 을 정의 했듯, T 타입을 정의한다음 F 에 대입한으로 Tee 를 정의할 수 있다.
  - Tee 를 생성하는 zipWith 라는 함수가 있다.
- 15.3.5 Sinks for output processing 
  - Sink 는 Process 의 output 을 표현하는 타입으로, (O) -> Process<F, Unit> 함수를 emit 하는 process 타입이다.
  - Sink 를 생성하는 fileW 라는 함수가 있다.
- 15.3.6 Hiding effects in effectful channels
  - Channel 은 순수한 pipeline 이 어떤 단계에서 I/O 를 수행해야 하는걸 표현할 때 유용하다. 
  - Channel<ForIO, (Connection) -> PreparedStatement, Map<String, Any>> 를 생성하는 query 와 같은 함수가 있다.
- 15.3.7 Dynamic resource allocation
  - Process 를 사용하면, 안전하게 동적으로 자원을 사용할 수 있다. 동적으로 자원을 할당한다는 것은 필요할때 필요한 만큼만 사용한다는 것을 의미한다.
   
Stream 이 의미하는 범위는 꽤 넓다. Process 는 데이터의 처리과정을 추상화 한 것이다. Process 들이 모이면 그것 또한 Stream 이라 할 수 있다.
우리는 stream transducer 를 데이터 스트림을 소비하고 변환하는 동안 조심스러운 지점에서 대기하고, 출력하고, 중지할 수 있는 state 머신으로 모델링 할 수 있다.
Stream transducers 는 processors 다.

