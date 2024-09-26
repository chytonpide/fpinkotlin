- 모나드는 아래의 법칙을 만족하는 추상적인 타입 클래스이다.
  - Left Identity
  - Right Identity
  - Associativity
- 모나드는 Higher-Kinded Types 이다.
  - Higher-Kinded Types 은 제네릭 타입에서 더 나아가, 타입 생성자를 매개변수로 받는 타입이다.
  - ex) Monad<List<String>>
- FP 에서는 어떤 효과를 모나드로 모델링 할 수 있다. 
  - 구체적인 모나드 타입에 의존하지 않고 함수나 연산을 정의할 수 있다.
- 코틀린은 Higher-Kinded Types 을 지원하지 않아서 이것을 Kind 로 구현한다.
- 효과를 모나드로 추상화 할 수 있다.
  - side effect 를 가지는 연산을 순수 함수 형식으로 다룰 수 있다.
  - 얘를 들어 IO작업을 모나드로 추상화 하면 아래와 같은 장점이 있다.
    - 지연실행
    - 조합: IO 작업을 컴비네이터를 사용해서 결합 시킬 수 있다.
    - 오류의 일관된 처리: FP 형식으로 처리가능
  - ex )
```kotlin
import arrow.fx.coroutines.*
import java.io.File

fun readFile(path: String): IO<String> = IO {
    File(path).readText()
}

fun writeFile(path: String, content: String): IO<Unit> = IO {
    File(path).writeText(content)
}

fun processContent(content: String): IO<String> = IO {
    content.uppercase()
}

val program: IO<Unit> = IO.fx {
    val content = !readFile("input.txt")
    val processedContent = !processContent(content)
    !writeFile("output.txt", processedContent)
}

val safeProgram: IO<Unit> = program.handleError { throwable ->
    IO { println("오류 발생: ${throwable.message}") }
}

fun main() {
    safeProgram.unsafeRunSync()
}

```
- 의미있는 컨택스트 내에서 어떤 상태를 관찰 할 수 있다면 그것은 side effect 이다.
FP 에서는 효과를 평가하지 않고, 효과를 기술하는 값을 반환하도록 한다. 이것으로 side effect 를 일으키지 않고 효과를 표현할 수 있다.
  - ex)
```kotlin
data class Console<A>(val run: () -> A)

// 순수 함수: 효과를 기술하는 값을 반환
fun greet(name: String): Console<Unit> {
    return Console { println("Hello, $name!") }
}

fun main() {
  val greeting = greet("Alice")
  greeting.run()  // 여기서 실제로 콘솔에 출력됨
}
```
  - 만약 Console 이 Monad 라면 flatMap 같은 컴비네이터로 연산을 조합 할 수 있다. 