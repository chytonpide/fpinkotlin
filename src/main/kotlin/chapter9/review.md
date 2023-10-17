# 9장 복기
## 학습목표
- FP 에서 사용되는 primitive, algebra 용어 개념 이해하기
- 각각의 함수들이 어떤 역할을 하는지 이해하기
- 각가의 함수들이 어떻게 조합되며 조합되면 어떤 기능을 하는지 이해하기

# 9 Parser combinator
- 이 장에서 배우는 것들
  - An algebraic design approach to libraries
  - Primitives vs higher-level combinators
  - Using combinators to achieve design goal
  - Improving library ease of use with syntactic sugar
  - Postponing combinator implementation by first focusing on algebra design
- algebraic design: 먼저 연관된 법칙들과 함께 인터페이스를 설계하고, 이것들의 조합이 데이터 타입의 표상에 대한 우리의 선택을 가이드 하도록 하는것.
- Index
  - 9.1 Designing an algebra
    - A parser to recognize single characters
    - A parser to recognize entire strings
    - A parser to recognize repetition
  - 9.2 One possible approach to designing an algebra
    - Counting character repetition
    - Slicing and nonempty repetition
  - 9.3 Handling context sensitivity
  - 9.4 Writing a JSON parser
    - Defining expectations of a JSON parser
    - Reviewing ths JSON format
    - A JSON parser
  - 9.5 Surfacing errors through reporting
    - First attempt at representing errors
    - Accumulating errors through error nesting
    - Controlling branching and backtracking
  - 9.6 Implementing the algebra
    - Building up the algebra implementation gradually
    - Sequencing parsers after each other
    - Capturing error messages through labeling parsers
    - Recovering from error conditions and backtracking over them
    - Propagating state through context-sensitive parsers
  - 9.7 Conclusion

## 9.1 Designing an algebra
- algebra: 함수들의 집합, 함수들 간의 관계를 명시한는 법칙들과 함께 데이터 타입들에 걸쳐서 동작하는,
- parsing 은 어떤 컴비네이터들이 요구되는지 상상하기 쉽기때문에 algebraic design 이 유요하다. representation 의 결정을 미루면서 구체적인 목표를 계속 추적 할 수 있다.
### 9.1.1 A parser to recognize single characters
- 반복되는 문자에 대한 파싱으로 부터 algebra 를 발견하는 것은 거대한 디테일을 무시한체 문제의 핵심에 집중하도록 한다.
- `fun char(c: Char): Parser<Char>` 특정 문자 하나를 파싱한느 파서를 생성해 난다. 
- `fun <A> run(p: Parser<A>, input: String): Either<PE, A>` 파서의 인터프리터, 파싱 결과와 에러를 나타낸다.
- 이 단계에서 표상이나 구현 디테일에 대해서 가능한 무지한 상태로 인터페이스를 특정하고 있다.
```kotlin
interface Parsers<PE> {
 
    interface Parser<A>
 
    fun char(c: Char): Parser<Char>
 
    fun <A> run(p: Parser<A>, input: String): Either<PE, A>
 
}
```
- 이제 다음과 같은 법칙을 만족시켜야 한다. `run(char(c), c.toString()) == Right(c)`
### 9.1.2 A parser to recognize entire strings
- 연속된 스트링을 인지하기 위해서 다음과 같은 함수가 필요하고 그 함수 또한 다음과 같은 법칙을 만족시켜야 한다.
- `fun string(s: String): Parser<String>`
- `run(string(s), s) == Right(s)
- 둘중 하나를 파싱 할 수 있는 파서를 생성하는 함수를 다음과 같이 정의할 수 있다.
- `fun <A> or(pa: Parser<A>, pb: Parser<A>): Parser<A>`
- 다음과 같은 infix 함수로 "문자열" or "문자열"로 string 파서를 or 로 함성한 Parser<String> 을 만들 수 있다.
- `infix fun String.or(other: String): Parser<String> = or(string(this), string(other))`
- or 함수는 다음과 같은 법칙을 만족시켜야 한다.
- `run("abra" or "cadabra", "abra") == Right("abra")`
### 9.1.3 A parser to recognize repetition
- 반복을 인지하는 파서를 만드는 함수를 다음과 같이 정의할 수 있다.
- fun <A> listOfN(n: Int, p: Parser<A>): Parser<List<A>>
- 여기까지 필요한 컴비네이터 들을 축척했지만 우리의 algebra 를 primitives 의 최소 세트로 개선하지 않았다. 그리고 더 일반적인 법칙에 대해서도 이야기 하지 않았다.
- THE ADVANTAGES OF ALGEBRAIC DESIGN: algebraic design 을 할 때 데이터 타입의 표상은 문제가 되지 않는다. 
  여기에는 그들 자신 내부의 표상 보다는 다른 타입과의 관계에 의해서 타입의 의미가 주어진다는, 카테고리 이론의 관점과 연관되어 있다.
- listOfN 함수는 다음과 같은 법칙을 만족시켜야 한다.
- `run(listOfN(3, "ab" or "cad"), "ababab") == Right("ababab")`

## 9.2 One possible approach to designing an algebra
### 9.2.1 Counting character repetition
- 0 또는 0 이상의 'a' 의 반복을 인지해서 그 문자의 수를 되돌려주는 파서를 구현해보자. 
- `fun <A> many(pa: Parser<A>): Parser<List<A>>` 와 같은 함수를 생각 할 수 있지만 우리가 원하는건 Parser<Int> 다.
  many 가 Parser<Int> 를 되돌려 주도록 할 수도 있지만, 그건 너무 구체적이다 대신에 아래와 같은 map 을 생각해 낼 수 있다.
- `fun <A, B> map(pa: Parser<A>, f: (A) -> B): Parser<B>`
- `map(many(char('a'))) { it.size }` 으로 우리가 원하는 파서를 생성할 수 있다.
- `val numA: Parser<Int> = char('a').many().map { it.size }` 으로 상수에 할당 하여 사용할 수 있고 numA 는 아래 법칙을 만족시켜야 한다.
- `run(numA, "aaa") == Right(3)`, `run(numA, "b") == Right(0)`
- 우리는 map 함수가 structure preserving 해야 한다고 예상함으로 다음과 같은 법칙을 만족시켜야 한다.
- `map(p) { a -> a } == p` 
- 이 법칙을 documentation 하는 방법도 있지만, property-based testing library 를 사용해서 법칙들이 실행가능 하게 할 수 있다.
```kotlin
object ParseError
 
abstract class Laws : Parsers<ParseError> {
    private fun <A> equal(
        p1: Parser<A>,
        p2: Parser<A>,
        i: Gen<String>
    ): Prop =
        forAll(i) { s -> run(p1, s) == run(p2, s) }
 
    fun <A> mapLaw(p: Parser<A>, i: Gen<String>): Prop =
        equal(p, p.map { a -> a }, i)
}
```
- 이렇게 함으로써 우리의 combinator 가 특정한 법칙을 만족하는지 확인 할 수 있는 수단이 생겼다. 새로운 법칙을 발견할때 마다 Laws 클래스에 쓰고 검증할 수 있다.
- `fun char(c: Char): Parser<Char> = string(c.toString()).map { it[0] }` 만약 우리가 strings 을 primitive 로 정의한다면 map 을 사용해서 char 도 정의할 수 있다.
- 비슷하게 항상 값 a 와 함께 성공하는 combinator 를 정의 할 수 있다. 
- `fun <A> succeed(a: A): Parser<A> = string("").map { a }` 여기서 a 는 Parser 가 결과로 리턴할 값이고 이 파서는 항상 a 를 리턴하고 다음과 같은 법칙을 만족해야 한다.
- `run(succeed(a), s) == Right(a)`
### 9.2.2 Slicing and nonempty repetition
- 시험된 input string 의 일부가 무엇인지 보는 파서가 있으면 좋을 것 같다.  
- `fun <A> slice(pa: Parser<A>): Parser<String>` 를 정의할 수 있고 다음과 같이 동작해야 한다.
- `run(slice(('a' or 'b').many()), "aaba") == Right("aaba")` 이는 many 에 의해서 누산된 list 를 무시하고 parser 에 의해 매치되는 input string 의 일부를 결과로 돌려준다.
- `char('a').many().slice().map { it.length }` 를 정의할 수 있고 이는 'a' 의 문자의 개수를 센다. 이 표현식에서 'a'의 개수를 셀 때 
  List 에서 size 를 도출하는게 아니라 String.length 를 사용한다. (String.length 는 계산하는데 constant time 이 필요하고, List.size() 는 길이에 따라 가변적이다.)
- 1 개 또는 그 이상의 'a' 를 인지하는 parser 가 필요하면 어떻게 해야 할까 일단 아래와 같은 many1 combinator 를 정의 할 수 있을 것이다.
- `fun <A> many1(p: Parser<A>): Parser<List<A>>`
- many1(p) 는 사실 p 와 그에 뒤따라오는 many(p) 과 같다. 그래 하나의 파서에 이어서(만약 첫번째 파서가 성공한다면) 또다른 파서를 실행하는 방법이 필요하다.
- `fun <A, B> product(pa: Parser<A>, pb: Parser<B>): Parser<Pair<A, B>>`
- 9.1) implement map2, many1
  - override fun <A, B, C> map2(pa: Parser<A>, pb: () -> Parser<B>, f: (A, B) -> C): Parser<C> = (pa product pb).map { (a, b) -> f(a, b) }  
  - override fun <A> many1(p: Parser<A>): Parser<List<A>> = map2(p, p.many()) { a:A, al:List<A> -> listOf(a) + al} 
- 9.2) product 를 특정하는 법칙을 도출하라
  - product 는 결합 법칙을 만족해야 한다. 전체 수식에서 연산자로 이어지는 피연산자의 순서가 같다면, 일부 수식을 먼저 계산하도록 하는 수식과 그 결과가 같다.
  - (a product b) product c == a product (b product c)
- 9.3) many 를 or, map2, succeed 를 이용해 구현하라
  - fun <A> many(pa: Parser<A>): Parser<List<A>> = map2(pa, many(pa)) { a, la -> listOf(a) + la } or succeed(emptyList())
- 9.4) listOfN 을 map2 와 succeed 를 사용해 구현하라
  - fun <A> listOfN(n: Int, pa: Parser<A>): Parser<List<A>> =
- 지금의 product 를 사용하는 many 의 구현은, map2 (내부적으로 product) 에서 다시 many 를 recursive 하게 call 함으로, 
  many 를 호출할 때, 이것이 무한반복된다. (run 을 통해 parsing 이 실행될 때가 아니라 many 가 호출되는 즉시, 무한 반복이 발생한다.)
  따라서 map2, product 에서 두번째 인자를 non-strict 하게 바꿀 필요가 있다. 
```kotlin
fun <A, B> product(
    pa: Parser<A>,
    pb: () -> Parser<B>
): Parser<Pair<A, B>> = TODO()

fun <A, B, C> map2(
    pa: Parser<A>,
    pb: () -> Parser<B>,
    f: (A, B) -> C
): Parser<C> =
    product(pa, pb).map { (a, b) -> f(a, b) }
```
- 9.5) defer 를 구현하고 기존 combinator 가 어떻게 바뀌는지 봐라 
  - fun <A> defer(pa: () -> Parser<A>): Parser<A> = pa()
  - map2(pa, defer({ many(pa) })) { a, la -> listOf(a) + la } or succeed(emptyList())
    - 인수로 전달 되는게 expression 이면 실행되지 않는다. nested 되는 경우도 마찬가지다. defer({ many(pa) }) 는 바로 evaluation 되지 않는다.   
- or 도 마찬가지로 non-strict version 을 구현 할 수 있다. 
- `fun <A> or(pa: Parser<A>, pb: Parser<A>): Parser<A>`
## 9.3 Handling context sensitivity
- 지금까지 구현한 primitive 들로는 "0", "1a", "2aa", "4aaaa" 같은 숫자가 뒤의 a 의 개수를 나타내는 context-sensitive grammar 를 parsing 할 수 없다.
  왜냐하면 두번째 parser 의 동작이 첫번째 parser 의 결과에 의존하기 때문이다.
- flatMap primitive 를 사용하면 이전 parser 의 결과에 의존하는 parser 의 연결을 만들 수 있다.
- 9.6) 숫자가 뒤의 a 의 개수를 나타내는 문법을 parsing 하는 parser 를 만들어라
  - 문법이 맞는지 확인한다. 결과로는 문자의 수만 돌려준다. 
```kotlin
val parser: Parser<Int> = regex("[0-9]+")
  .flatMap { digit: String ->
    val reps = digit.toInt()
    listOfN(reps, char('a')).map { _ -> reps }
  }
```
- 9.7) Implement product and map2 in terms of flatMap and map.
```kotlin
fun product(pa: Parser<A>, pb: Parser<B>): Parser<Pair<A, B>> = pa.flatMap({a -> pb.map { b -> a to b }}) 
fun <A, B, C> map2(pa: Parser<A>, pb: () -> Parser<B>, f: (A, B) -> C): Parser<C> = pa.flatMap({a -> pb.map { b -> f(a, b)}})
```
- 9.8) map is no longer primitive. Express it in terms of flatMap and/or other combinators.
```kotlin
fun <A, B> map(pa: Parser<A>, f: (A) -> B): Parser<B> = pa.flatMap{ a -> succeed(f(a)) }
```
- 더 일반적인 flatMap 을 도입 함으로써 primitives 는 다음과 같이 여섯개가 되었다. string, regex, slice, succeed, or, flatMap
- flatMap 의 도입으로 context-sensitive grammar 를 파싱할 수 있게 되었고, 이는 C++ 같은 복잡한 문법도 parsing 할 수 있다.
- 지금까지 우리는 이 primitives 의 구현에 거의 시간을 쓰지 않고 대신에 Parsers 인터페이스에 추상적인 정의들을 정의하면서 우리의 algebra 를 도출해 냈다.
  이 접근 법을 고수해서 이 primitives 구현을 가능한 뒤로 미뤄보자.

## 9.4 Writing a JSON parser
### 9.4.1 Defining expectations of a JSON parser
- JSONParser 는 다음과 같은 형태 일 것이다.
```kotlin
object JSONParser : ParsersImpl<ParseError>() {
    val jsonParser: Parser<JSON> = TODO()
}
```
- FP 의 경우, algebra 를 정의하고, 구현 정의에 앞서 그것의 expressiveness 를 탐구하는 것은 일반적이다.
  이후에 error reporting 기능을 추가하고, 우리의 parser 타입의 구체적인 구현을 도출할 것이지만 이것은 여기서 구현하는 json parser 의 구현과 완전히 독립적이다.
### 9.4.2 Reviewing the JSON format
- JSON 을 parsing 하려면 JSON Data type 이 필요하다.
```kotlin
sealed class JSON {
    object JNull : JSON()
    data class JNumber(val get: Double) : JSON()
    data class JString(val get: String) : JSON()
    data class JBoolean(val get: Boolean) : JSON()
    data class JArray(val get: List<JSON>) : JSON()
    data class JObject(val get: Map<String, JSON>) : JSON()
}
```
### 9.4.3 JSON parser
- 9.9) JSON parser 를 구현하라.
  - 내 접근법
```kotlin

abstract class Parsers<PE> {
  infix fun <A,B> Parser<Pair<out A, out B>.left() Parser<A> =  this..map { pair -> pari.first}
  infix fun <A,B> Parser<Pair<out A, out B>.right() Parser<B> =  this..map { pair -> pari.second}
}

val JValueParser = regex("큰 따옴표 안의 택스트") or regex("숫자").map { value -> 
        if(numberRegex.containsMatchIn(value)) {
            JNumber(pair.second)
        } else if(stringRegex.continsMatchIn(Value)) {
            JString(pair.second)
        } else {
            JNull
        }
}


val JNameValueParser = (regex("큰 따옴표 안의 택스트") product char(":").map { pair -> pair.first} product JValueParser).map { pair -> mapOf(pair.first to pair.second) }
val jsonParser = (((char('{') product (JNameValueParser product char(',').left()).many()).right().map { list -> list.foldMap()} product JNameValueParser).map { pair -> JObject(pair.first + pair.second)}) product char('}').left(); 

run(jsonParser, "{\"name\":\"kang\", \"age\": 36, \"job\": \"engineer\"}") = Right(JObject(mapOf("name" to JString("kang"), "age" to JNumber(36))))
```
- combinator 를 먼저 정의하여 parser 도출
```kotlin
abstract class Parsers<PE> {
  // ...
  // combinator
  internal abstract fun <A> skipR(
    pa: Parser<A>,
    ps: Parser<String>
  ): Parser<A>

  internal abstract fun <B> skipL(
    ps: Parser<String>,
    pb: Parser<B>
  ): Parser<B>

  internal abstract fun <A> sep(
    p1: Parser<A>,
    p2: Parser<String>
  ): Parser<List<A>>

  internal abstract fun <A> surround(
    start: Parser<String>,
    stop: Parser<String>,
    p: Parser<A>
  ): Parser<A>
}
abstract class JSONParsers : ParsersDsl<ParseError>() {
  // ...
  fun array(): Parser<JArray> =
    surround("[".sp, "]".sp,
      (value sep ",".sp).map { vs -> JArray(vs) })

  fun obj(): Parser<JObject> =
    surround("{".sp, "}".sp,
      (keyval sep ",".sp).map { kvs -> JObject(kvs.toMap()) })

  fun <A> root(p: Parser<A>): Parser<A> = p skipR eof

  val jsonParser: Parser<JSON> =
    root(whitespace skipL (obj() or array()))
}
```

## 9.5 Surfacing errors through reporting
- 지금까지의 combinators 들은 grammar 가 무엇인지, parsing 이 성공한다면 성공했을 경우 어떻게 처리하는지에 대해서만 특정했다.
  이 시점에서 디자인이 완료된 것으로 간주하고 구현으로 넘어간다면 error 레포팅에 대해서 몇가지 임의이 결정을 내려야 하고 이 결정들은 적절하지 않을 가능성이 높다. 
  따라서 error 레포팅에 대해서도 algebraic design 어프로치를 사용해야 한다. 
  이 장에서 parser 로 하여금 reported 되는 error 들을 표현하기 위한 combinator 들을 정의할 것이다.
### 9.5.1 First attempt at representin errors
- 다음과 같이 에러가 발상했을 때 파서가 설정한 메세지를 되돌려줄 수 있도록 명백한 컴비네이터를 정의할 수 있다.
```kotlin
fun <A> tag(msg: String, p: Parser<A>): Parser<A>
```
- 하지만 우리는 우리의 파서가 어디서 error 가 발생했는지도 말해주기를 원한다. 따라서 Location 이라는 개념을 우리의 algebra 에 실험적으로 적용해 보자.
```kotlin
data class Location(val input: String, val offset: Int = 0) {
 
    private val slice by lazy { input.slice(0..offset + 1) }
 
    val line by lazy { slice.count { it == '\n' } + 1 }
 
    val column by lazy {
        when (val n = slice.lastIndexOf('\n')) {
            -1 -> offset + 1
            else -> offset - n
        }
    }
}
 
fun errorLocation(e: ParseError): Location
 
fun errorMessage(e: ParseError): String
```
### 9.5.2 Accumulating errors through error nesting
- tag 를 사용할때 우리는 조금더 문맥에 관한 정보를 가져야 한다. 만약 어디에서 부터 잘못되었는지 알 수 있다면 아주 유용할 것이다.
  만약 에러 메세지가 "cAdabra" 의 파싱에서 A 가 예상치 못한 문자라고 말해준다면 이상적일 것이다. 따라서 일차원적인 에러 레포팅은 충분하지 않을 수 있고,
  tag 를 nesting 하는 방법이 필요하다. scope 라는 컴비네이터는 p에 첨부된 tag 를 유지하면서 p 가 실패할 경우 추가정보를 더한다. 
```kotlin
fun <A> scope(msg: String, p: Parser<A>): Parser<A>
```
- 하나의 Location 과 String 메세지가 아니라, List<Pari<Location, String>> 를 가져야 한다. 따라서 ParseError 를 다음과 같의 정의 해야 한다.
```kotlin
data class ParseError(val stack: List<Pair<Location, String>>)
```
- 이제 복수의 errors 가 있을때 scope 가 어떻게 동작해야하는지 특정할 수 있다.
  만약 run(p, s) 가 Left(e1) 이면,
  run(scope(msg, p), s)) 는 Left(e2) 이고, e2.stack.head 는 msg 이고, e2.stack.tail 는 e1 이다.
- 우리는 reporting 목적을 위하여 에러가 관련된 모든 정보를 포함하기를 원했다. 이제 ParseError 는 거의 모든 목적에 충분해 보임으로 이것을 
  run 함수의 return 타입으로 사용하는 구체적인 표상으로 취하도록 한다.
```kotlin
fun <A> run(p: Parser<A>, input: String): Either<ParseError, A>
```
### 9.5.3 Controlling branching and backtracking
- or 로 조합된 parsers 중에 어는 parser 의 error 를 리포팅 할지 결정해야 하는 한가지 이슈가 남았다.
  이것을 global convention 으로 다루지 않고, program 이 선택하게 할 것이다. 
- 우리는 특정 파싱 프런치에 언제 commit 하는지 프로그래머가 지시할 수 있는 primitive 가 필요하다.
  p1 or p2 는 "p1 을 시도하고 p1 이 실패하면 p2 를 시도 하라" 라는 의미에서
  "input 에 대해서 p1 을 시도하고 만약 실패하면 uncommitted state 에서 p2 를 같은 input 으로 시도하고 실패를 리포트하라" 라는 의미가 될 것이다.
- 기본적으로 모든 parsers 들이 commit 하도록 하고 committing 을 delay 하는 다음과 같은 combinator 를 정의 할 수 있다.
```kotlin
fun <A> attempt(p: Parser<A>): Parser<A>
```
- 만약 우리가 p1 과 p2 가 실패 했을 경우 p2 가 두 가지에 대한 에러를 포함한다면 아래 등식은 성립하지 않는다. 하지만 attempt 는 실패한 다면, commit 을 revert 하기 때문에
아래 등식이 성립한다.
```kotlin
attempt(p1.flatMap { _ -> fail }) or p2 == p2
```
- 9.10) or 체인에서 어떤 error 가 리포팅 되어야 하는지 특정하는 편리한 primitive 를 생각 해 봐라.
  - `fun <A> furthest(pa: Parser<A>): Parser<A>` `fun <A> latest(pa: Parser<A>): Parser<A>` 중첩된 에러를 제거한다?
- 우리는 아직도 algebra 의 구현을 작성하지 않았다. 구현이 없음에도 이 프로세스는 우리의 combinators 들이 우리의 라이브러리 유저에게 잘 정의된 인터페이스를 제공한다는 것을 확신하게 해 왔다.
  이 프로세스는 하위 구현에 대해 그들로하여금 올바른 정보를 전달하는 길을 제공해야 한다.
  이 프로세스는 이제 우리가 규정해 온 법칙들을 만족하는 방식으로 이 정보드를 해석하는 구현에 이를 것이다.

## 9.6 Implementing the algebra
- Parser 지금 그냥 type token 이다. Parsers 인터페이스의 구현에서 사용할 수 있도록 Parser 의 representation 을 먼저 정의해야 한다. 
  도출된 법칙들을 만족하려면 Parser 는 purely functional representation 이어야 할 것이다. 
### 9.6.1 Building up the algebra implementation gradually
- Parser 를 run function 의 구현이라고 가정할 수 있다.
```kotlin
typealias Parser<A> = (String) -> Either<ParseError, A>
```
```kotlin
override fun string(s: String): Parser<String> =
  { input: String ->
        if(input.startsWith(s)) {
            Rgiht(s)
        } else {
            Left(Location(input).toError("Expected: $s"))
        }
  }
private fun Location.toError(msg: String) = 
    ParserError(listOf(this to message))
```
### 9.6.2 Sequencing parsers after each other
- Parser 의 현재의 representation 으로는 "abra" product "cadabra" 를 표현할 수 없다.
- Parser 로 하여금 글자가 몇개나 소비되었는지 나타낼 수 있게 해야 하는데, Location 이 input string 과 그의 offset 을 가지고 있는걸 고려하면 이것은 쉬워 보인다.
```kotlin
typealias Parser<A> = (Location) -> Result<A>
sealed class Result<out A>
data class Success<out A>(val a:A, val consumed:Int): Result<A>()
data class Failure(val get: ParserError): Result<Nothing>()
```
- 우리는 Either 의 대한으로 조금 더 풍부한 데이터 타입인 Result 를 도입했다.
- 이 타입은 파서가 진정으로 어때야 하는지에 대한 정수를 향해가기 시작했다. 이것은 실패할 수 있는 state action 의 한 종류로,
  input state 를 받고, 성공시에 값과 함께 어떻게 state 가 update 되어야 하는지 컨트롤 하기 위한 충분한 정보를 되돌려 준다.
- Parser 가 state action 이라는 이해는, 우리에게 우리가 지금까지 규정한 법칙과 컴비네이터를 지원하는 representation 을 모양짓는 방법을 준다.
- 9.11) 현재 representation 의 Parser 의 string, regex, succeed, slice 를 구현하라.
  - 각각의 combinator 들을 state 를 직접 변경시키지 않는다. Success 에 parsing 결과와 그 결과를 얻기 위해 몇 글자나 commit 했는지를 기록 해서 돌려준다.
    실패 시에는 state 를 활요해서 error 정보를 담고 있는 새로운 state(Location) 을 생성하고 그것을 ParserError 의 형태로 Failure 에 기록해서 돌려준다. 
  
### 9.6.3 Capturing error messages through labeling parsers
- scope 컴비네이터를 통해서 우리는 새로운 메세지를 ParserError 스택에 push 하고 싶다.
```kotlin
// data class ParseError(val stack: List<Pair<Location, String>>)

fun ParseError.push(loc: Location, msg: String): ParseError =
    this.copy(stack = (loc to msg) cons this.stack) //stack 이니깐 최신의 것이 앞쪽에 위치한다.
```
- push 와 mapError　컴비네이터를 이용해서 scope 를 구현 할 수 있다.
```kotlin
fun <A> scope(msg: String, pa: Parser<A>): Parser<A> =
    { state -> pa(state).mapError { pe -> pe.push(state, msg) } }
```
```kotlin
fun <A> Result<A>.mapError(f: (ParseError) -> ParseError): Result<A> =
    when (this) {
        is Success -> this
        is Failure -> Failure(f(this.get))
    }
```
- `scope(msg1, a product scope(msg2, b))` 에서 b 파서가 실패할 경우 stack 의 첫번째(list 의 첫번째) error 는 msg1 이 될 것이고,
  그 뒤를 a 가 만들어 내는 error 가 되고, 그 다음 msg2, 마지막으로 b 파서가 만들어내는 error 가 될 것이다.
- tag 를 기존 error 의 메세지를 변경하도록 tag 를 재정의하고, ParserError 상에 tag helper function 을 정의 했다.
```kotlin
fun <A> tag(msg: String, pa: Parser<A>): Parser<A> =
    { state ->
        pa(state).mapError { pe ->
            pe.tag(msg)
        }
    }
```
- tag helper function 을 stack 에서 가장 최근의 location 잘라내어서 msg 와 연결시키는 함수로 설계 했다.
```kotlin
fun ParseError.tag(msg: String): ParseError {
 
    val latest = this.stack.lastOrNone()
 
    val latestLocation = latest.map { it.first }
 
    return ParseError(latestLocation.map { it to msg }.toList())
}
```
- 9.12) 9.11 과 답 같음

### 9.6.4 Recovering from error conditions and backtracking over them
- or 의 정의: it should run the first parser, and if that fails in an uncommitted state, it should run the second parser on the same input.
  uncommitted state 를 위해서 Failure 타입에 isCommitted 파라미터를 추가 할 수 있다.
```kotlin
data class Failure(
    val get: ParseError,
    val isCommitted: Boolean
) : Result<Nothing>()
```
- attempt 의 구현은 이 새로운 정보 위에서 그려지고, failures 가 일어나면 commitment 를 캔슬 한다.
```kotlin
fun <A> attempt(p: Parser<A>): Parser<A> = { s -> p(s).uncommit() }
 
fun <A> Result<A>.uncommit(): Result<A> =
    when (this) {
        is Failure ->
            if (this.isCommitted)
                Failure(this.get, false) // isCommitted 를 false 로 덮어 씌움
            else this
        is Success -> this
    }
```
- or 을 다음과 같이 구현 할 수 있다.
```kotlin
fun <A> or(pa: Parser<A>, pb: () -> Parser<A>): Parser<A> =
  { state ->
    when (val result = pa(state)) {
      is Failure -> 
        if(result.isCommitted)
            result
        else 
            pb()(state)    
      is Success -> result
  }
```
### 9.6.5 Propagating state through context-sensitive parsers
- 마지막으로 flatMap 을 구현해 보자. 
```kotlin
fun <A, B> flatMap(pa: Parser<A>, f: (A) -> Parser<B>): Parser<B> =
  { state ->
    when(val result = pa(state)) {
      is Success -> {
        val newState = state.advanceBy(result.consumed) // state 를 업데이트 한다. 
        val parsingResult = result.a
        val pb = f(parsingResult)
        val resultb = pb(newState).addCommit(result.consumed != 0) // 업데이트 된 state 로 pb 를 실행한후,  
        val resultb2 = resultb.advancedSuccess(result.consumed) // Success 는 소비된 글자수를 가져야한다. 
      }
     is Failure -> result
    }
  }
```
- state 의 업데이트는 flatMap 에서 이루어 진다. product 도 map2 도 flatMap 을 사용한다 
- 9.13) run 을 구현하라
```kotlin
abstract class ParsersImpl<PE> : Parsers<PE>() {
    //tag::init[]
    override fun <A> run(p: Parser<A>, input: String): Result<A> =
        p(Location(input))
    //end::init[]
}
```
- 9.14) ParserError 을 사람을 위해 포맷하는 좋은 방법을 도출하라, 
  중요한 통찰 중 하나는 error 을 문자열로 표시할 때 정확한 위치에 첨부된 태그를 결합하거나 그룹화 하는 경향이 있다는 것이다.
```kotlin
data class ParseError(val stack: List<Pair<Location, String>> = emptyList()) {

    fun push(loc: Location, msg: String): ParseError = this.copy(stack = listOf(loc to msg) + this.stack)

    fun label(s: String): ParseError =
        ParseError(latestLoc()
            .map { it to s }
            .toList())

    fun latest(): Option<Pair<Location, String>> = this.stack.lastOrNone()
  
    fun latestLoc(): Option<Location> = latest().map { it.first }
}
```
- 우리는 파서 컴비네이터 라이브러리가 이 챕터에서 챙겨야할 가장 중요한게 아니라도 이것을 계속 해서 발전시켜 왔다. 
  그러나 이 과정은 algebra-first library design 의 모든 것을 묘사한다.

## 9.7 Conclusion
- Algebraic library design 은 먼저 연관된 법칙들과 함께 인터페이스를 수립한다. 그리고 나서 구현으로 이끈다.
- parser combinator library 는 functional library design 에 대한 동기를 제공하며, algebraic design approach 에 잘 맞는다.
- Primitives 는 다른 것들에 의존하지 않는 단순한 컴비네이터 이며, 그들은 더 고차원의 컴비네이터를 위한 빌딩 블록을 제공한다.
- Algebraic design 은 먼저 Primitive 의 발명을 격려하고 이는 더 복잡한 컴비네이터들의 발견이 따라오도록 허용한다.
- 컴비네이터가 상태를 전달하면서 연결할 수 있을 때, 컴비네이터는 context sensitive 하다고 할 수 있다.
- parser combinator 는 에러를 축적할 수 있고, 이것은 실패시에 에러 리포팅을 드러내게 한다.
- parser combinator 는 uncommitted state 로 실패할 수 있고, 이것은 철회와 에러로부터의 복구를 가능하게 한다.
- Algebra 와 함께 설계를 시작하는 것은 컴비네이터 들이 구현에 대한 정보를 명시하도록 한다. (Parser 의 representation Parser<A> = (Location) -> Result<A> 이 그렇게 결졍되었다.) 


## 요약
- algebra: 함수들 간의 관계를 명시한는 법칙들과 함께 데이터 타입들에 걸쳐서 동작하는 함수들의 집합.
- algebraic design: 먼저 연관된 법칙들과 함께 인터페이스를 설계하고, 이것들이 이것들은 만족해야하는 구현(표상)에 대한 정보를 제공하도록 하는 것.
- 9.1 Designing an algebra
  파서는 입력에 대해 기대되는 아웃풋이 분명하기 때문에 algebraic design approach 를 하기에 적합하다.
  파싱은 format 을 가진 어떤 문자열을 파싱을 통해서 자료구조로 만드는 것 
- 9.2 One possible approach to designing an algebra
  반복에 대한 파서를 생성하는 컴비네이터를 구현하고, 더 일반적인 map 컴비네이터를 도출, product 와 같은 여러 컴비네이터 도출
- 9.3 Handling context sensitivity 
  이전 파서의 결과가 이후 파서의 동작에 영향을 끼치는 context sensitivity parser 의 필요성 확인, 이를 위한 flatMap 컴비네이터 도출
  컴비네이터 들 중에 더 일반적인 것들을 primitive 로 정의
- 9.4 Writing a JSON parser
  JSON format 리뷰, JSON parser 가 돌려줄 JSON 타입을 정의한 다음, 구현이 없는 parser combinator library 의 interface 만으로 JSON parser 를 정의 할 수 있다.
- 9.5 Surfacing errors through reporting
  error 를 표현하는 가장 간단한 방법 부터, 위치정보를 가지고 error 를 누산 할 수 있게 ParserError 의 representation 을 변경했다.
  commitment 개념을 도입해서 backtracking (철회) 가 가능하도록 했다.
- 9.6 Implementing the algebra
  법칙을 만족 시킬 수 있는 가장 단순한 Parser 의 리프레젠테이션 (String) -> Either<ParseError, A> 부터 시작해서,
  parser 들을 연결 할 수 있다는 법칙을 만족 시키기 위해 상태가 필요하는걸 깨닫고 상태로 (앞에 파서가 몇글자나 소비했고 얼마나 남았는지) 
  error reporting 을 위해 정의했던 Location 타입을 정의하고 representation 을 (Location) -> Result<A> 로 변경했다.
  Success 또는 Failure 인 Result 타입을 정의하고 Failure 는 ParseError 를 가지도록 Success 는 parsing 결과 a 와 consumed 를 가지도록 했다.
  Failure 가 isCommitted 를 가지도록 함으로써 attempt 를 완전하게 정의했다. or 또한 Failure 의 isCommitted 를 활용하도록 정의해서 
  프로그래머로그 하여금 uncommitted 한 parsing 을 명시할 수 있도록 했다.     
  parser 의 모든 sequencing 은 flatMap 을 사용한다. 따라서 flatMap 에서 Success 의 consumed 를 사용해 state 를 변경하도록 했다.
  