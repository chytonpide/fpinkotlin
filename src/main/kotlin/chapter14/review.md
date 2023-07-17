# 14장 복기
## 학습목표
- 각 타입들의 관계에 대해서 이해하기
- 각 타입들이 어떤 역할을 가지는지 이해하기
- 레이피케이션이 어떤 흐름으로 이루어지는지 이해하기
- 레이피케이션을 통해서 어떤 개념들이 명시화되는지 이해하기

# 14 Local effects and mutable state
- 이 챕터에서 배우는 것
  - mutable state 의 관점에서의 referential transparency 의 정의
  - effect 의 typed scoping 을 통해서 local state 의 변경 숨기기
  - mutable state 를 encapsulate 하기 위한 DSL 의 개발
  - 프로그램 동작을 위한 algebra 와 interpreter 의 수립

## 14.1 State mutation is legal in pure function code
- An expression e is referentially transparent if for all programs p, 
  l occurrences of e in p can be replaced by the result of evaluating of e with affecting the meaning of p.
- A function f is pure if the expression f(x) is referentially transparent for all referentially transparent x.
- 어떠한 함수던지 local side effect (state mutation) 를 가지면서, 자신의 caller 에게는 pure 한 interface 만 노출하는 pure 한 function 이 될 수 있다.

## 14.2 A data type to enforce scoping of side effect
- local side effect 와 side effect 를 구분 짓는 타입을 정의한다음 우리가 의도치 않게 side effect 를 노출했을 경우 compiler 로 하여금 경고를 받게 할 것이다.
- IO action 은 나중에 실행될 때 side effect 를 가질 수도 있고, 가지지 않을 수도 있다. 따라서 진짜 I/O 와 같은 side effect 를 가지는 IO action 과 locally mutable state
  만 가지는 IO action 을 새로운 타입을 도입해 구분 할 것이다.

### 14.2.1 A domain-specific language for scoped mutation
- s 를 입력으로 받고, Pair<A, S> 를 돌려주는 State<S, A> 라는 타입을 이미 알고 있지만 여기서 우리가 정의할 타입에서는 state 를 전달하지 않는다.
  새로운 데이터 타입에서는 type S 로 마크된 토큰을 전달하면 그 타입은 (타입의 함수) 는 같은 타입 S 를 변경할 수 있는 권한을 얻게 될 것이다. 
- 우리의 새로운 데이터 타입은 아래의 두 invariant 가 지켜지지 않으면 컴파일 되지 않도록 할 것 이다.
  - If we hold a reference to a mutable object, then nothing should observe us mutating it from the outside
  - A mutable object should never be observed outside of the scope in which it was created.
- 우리가 읽기 List<Byte> view 를 호출자에게 반환한다고 해도, 우리의 스코프 내에서 array 를 변경하면 그게 view 에 반영되고, 그 변경은 호출자에게서 관찰된다.
  따라서 우리는 List<Byte> view 의 범위를 제한하고 우리가 이것과 연결된 array 를 변경할때 caller 가 이것에 대한 참조를 가지지 못하도록 해야한다.
- local effect 를 위한 새로운 모나드를 ST(state thread) 라고 부른다. State 모나드와 다른점은 run 메서드가 protected 라는 것이다.
  `typealias STOf<S, A> = arrow.Kind2<ForST, S, A>`
```kotlin
abstract class ST<S, A> internal constructor() : STOf<S, A> {
    // ST({a}) 의 형태로 호출됨. 외부에서는 constructor 를 호출 할 수 없음으로, 이 companion object 로만 객체를 생성할 수 있다.  
    companion object {
        operator fun <S, A> invoke(a: () -> A): ST<S, A> {
            val memo by lazy(a)
            return object : ST<S, A>() {
                override fun run(s: S) = memo to s
            }
        }
 
    }
  
    protected abstract fun run(s: S): Pair<A, S>
 
    fun <B> map(f: (A) -> B): ST<S, B> = object : ST<S, B>() {
        override fun run(s: S): Pair<B, S> {
            val (a, s1) = this@ST.run(s)
            return f(a) to s1
        }
    }
 
    fun <B> flatMap(f: (A) -> ST<S, B>): ST<S, B> = object : ST<S, B>() {
        override fun run(s: S): Pair<B, S> {
            val (a, s1) = this@ST.run(s)        
            return f(a).run(s1)
        }
    }
}
```
- S 가 state 를 mutate 할 수 있는 능력을 의미하기 때문에 run 메서드는 protected 이다. 
- ST의 구현 디테일보다는 mutable state 의 scope 를 제한하기 위해 타입 시스템을 사용하는 아이디어가 중요하다. 

### 14.2.2 An algebra of mutable references
- ST 모나드를 정의했다. ST 모나드의 첫번째 응용은 mutable reference 에 관한 DSL 이다.
- 이 언어는 몇가지 primitives 를 가지는 combinator library 로 구성되고
  변경 가능한 메모리 셀을 캡슐화하고 격리하는 이러한 참조에 대해 이야기하기 위해 이 언어는 아래와 같은 기본 명령어를 가져야 한다.
  - Allocate a new mutable cell
  - Write to a mutable cell
  - Read from a mutable cell
```kotlin
abstract class STRef<S, A> private constructor() {
    companion object {
        operator fun <S, A> invoke(a: A): ST<S, STRef<S, A>> = ST {
            object : STRef<S, A>() {
                override var cell: A = a
            }
        }
    }
 
    protected abstract var cell: A
 
    fun read(): ST<S, A> = ST {
        cell
    }
 
    fun write(a: A): ST<S, Unit> = object : ST<S, Unit>() {
        override fun run(s: S): Pair<Unit, S> {
            cell = a
            return Unit to s
        }
    }
}
```
- ST<S,A> type hides local mutations from observers within the STRef<S,A> type.
  - STRef<S, A>(a) 는 ST 의 constructor 에 cell 에 a 가 할당된 STRef<S, A> 전달해서 ST<S, STRef<S, A>> 를 생성한다.
  - STRef<S, A>.read() 는 ST 의 constructor cell 전달해서 ST<S, A> 를 생성한다.
  - STRef<S, A>.write(a) 는  cell 에 a 를 할당하는 run 을 override 한 ST 의 구현체 ST<S, Unit> 을 생성한다.  
- S 타입은 cell 의 타입이 아니다. 그렇다고 해도 만약 우리가 ST 액션 중 하나를 호출하고 실행하기 위해서는 S 타입의 값을 필요로 한다.
  S 는 cell 을 변경하거나 접근할 수 있는 일종의 보안 token 처럼 사용된다.
- STRef 의 constructor 는 private 임으로 우리는 초기값을 인수로 전달하여 STRef 의 companion object 의 invoke 로만 STRef 객체를 얻을 수 있다.
  생성되는 ST action 의 타입은 ST<S, STRef<S, A>> 이고, S 를 사용해서 run 을 해야하만 STRef 를 얻을 수 있다. 이때 STRef 와 ST 의 S 는 같다.

### 14.2.3 Running mutable state actions
- STRef 를 사용해서 mutable references 를 encapsulate 했지만 외부에서 ST<S, STRef<S, Int>> 를 실행시켜서 STRef<S, Int> 를 얻을 수 있다면
  STRef 안의 가변 변수를 외부에서 관찰 할 수 있고, 이것은 ST 의 invariant 를 위반한다.
  따라서 우리는 STRef 를 노출하는 ST<S, STRef<S, Int>> 를 실행할 수 없도록 보장하고 싶다.
- RunnableST 으로 ST 를 wrapping 하고, ST 의 실행 함수를 runST(st: RunnableST<A>): A 로 정의함으로써 local mutation 을 외부로 부터 완전히 감출 수 있고
  ST 의 invariant 의 위반을 type system 을 사용해 검출 할 수 있게 되었다. ST 는 side effect 가 없는 action 을 의미한다.

### 14.2.4 The mutable array represented as a data type for the ST monad
```kotlin
abstract class STArray<S, A> @PublishedApi internal constructor() {
 
    companion object {
        inline operator fun <S, reified A> invoke(
            sz: Int,
            v: A
        ): ST<S, STArray<S, A>> = ST {
            object : STArray<S, A>() {
                override val value = Array(sz) { v }
            }
        }
 
    }
 
    protected abstract val value: Array<A>
 
    val size: ST<S, Int> = ST { value.size }
 
    fun write(i: Int, a: A): ST<S, Unit> = object : ST<S, Unit>() {
        override fun run(s: S): Pair<Unit, S> {
            value[i] = a
            return Unit to s
        }
    }
 
    fun read(i: Int): ST<S, A> = ST { value[i] }
 
    fun freeze(): ST<S, List<A>> = ST { value.toList() }
 
}
```
- ex) 14.1 fun <S, A> STArray<S, A>.fill(xs: Map<Int, A>): ST<S, Unit> =
  - xs 를 fold 함 초기값으로 ST { Unit } 을 설정한다음 st.flatMap { write(k, v) } 으로 ST 를 변환시켜감
  - STArray<S, A> 의 extension method 이다. STArray<S, A> 를 가지고 있는 상태에서 사용하는 메서드임으로 결국 이것은 ST 내부에서 호출되는 함수이다. 
- inline 선언과 reified type A 를 사용해 함수를 정의하면 caller 에서 정의한 type parameter 에 함수 내부에서 접근 할 수 있다.

### 14.2.5 A purely functional in-place quicksort
- ex) 14.2 Write the purely functional versions of partition and qs.
  - partition 에서는 array 의 조작에서 STArray 를 사용하고, 일시적으로 값을 유지하기 위해 STRef 를 사용한다.
  - qs 는 sort 할 array 를 STArray 로 받고, 이것을 조작한 결과로 ST<S, Unit> 을 리턴한다. 이것들은 ST 내부에서 사용할 수 있는 함수다.
  - 결과 적으로 ST 를 사용하는 quicksort 는 아래와 같다.
```kotlin
fun quicksort(xs: List<Int>): List<Int> =
    if (xs.isEmpty()) xs else ST.runST(object : RunnableST<List<Int>> {
        override fun <S> invoke(): ST<S, List<Int>> =
            ST.fx {
                val arr = STArray.fromList<S, Int>(xs).bind()
                val size = arr.size.bind()
                qs(arr, 0, size - 1).bind()
                arr.freeze().bind()
            }
    })
```
- RunnableST 를 정의함으로써 함수안의 STRef 나 STArray 를 외부에 노출 할 수 없다.  
- 내부에서의 state transition 은 STRef 나 STArray 만 사용한다.
- quicksort 는 ST 를 사용해서 정의했기 때문에 side effect 가 없다.
- ex) 14.3 STMap 을 정의하라
  - STMap.fromMap
  - get
  - set
  
# 14.3 Purity is contextual
## 14.3.1 Definition by example
- constructor 는 memory 상에 유니크한 오브젝트를 생성한 다음 그것의 레퍼런스를 돌려주는 effect 를 가지고 있다.
- 거의 모든 컨택스트에서 프로그램은 === 같은 함수를 사용해서 레퍼런스를 관찰하지 않기 때문에 constructor 사용하는 프로그램이 side effect 가 없다고 말 할 수 있다.
- 기존
  - An expression e is referentially transparent 
    if for all programs p, all occurrences of e in p can be replaced by the result of evaluating e without affecting the meaning of p.
- 특정 프로그램으로 scoped 된 컨택스트를 포함하는 새로운 정의 
  - An expression e is referentially transparent with regard to a program p 
    if every occurrence of e in p can be replaced by the result of evaluating e without affecting the meaning of p.
  - kotlin 에서 evaluation 은 어떤 일반적인 형태로의 환산을 의미한다. 여기서 expression e 의 evaluation 을 val 에 할당 함으로 일반 적인 형태로 강제할 수 있다.
    `>>> val v = e`
  - we talk about referential transparency, it’s always regarding some context.
## 14.3.2 What counts as a side effect?
- The policy we should adopt is to track those effects that program correctness depends on.
- If a program relies on object reference equality, it would be nice to know that statically, too. 
  Static type information lets us know what kinds of effects are involved, 
  thereby letting us make educated decisions about whether they matter to us in a given context.
- 우리의 문맥에서 어떤 효과가 프로그램의 정확성과 관련이 있는지 판단하고 중요하다면 그것을 추적하는데 정적 타입 정보를 사용할 수 있게 하기 위해서
  A domain-specific language 를 정의해야 한다. ST나 IO 같은 타입을 정의하고 나아가 algebra 를 정의해야 한다.




# 지금까지 컨택스트
## 14 Local effects and mutable state
### 14.1 State mutation is legal in pure function code
- RT, Pure Function 의 정의를 확인함, local state mutation 이 pure function code 에서 허용됨을 확인 함.
### 14.2 A data type to enforce scoping of side effect
- IO action 은 나중에 실행될 때 side effect 를 가질 수도 있고, 가지지 않을 수도 있다. 따라서 진짜 I/O 와 같은 side effect 를 가지는 IO action 과 locally mutable state
  만 가지는 IO action 을 새로운 타입을 도입해 구분 할 것이다.
#### 14.2.1 A domain-specific language for scoped mutation
- 아래 두가지 invariants 가 지켜지지 않으면 compile 이 되지 않도록 하는 타입으로 ST (state thread) 를 정의 했다.  
  - If we hold a reference to a mutable object, then nothing should observe us mutating it from the outside
  - A mutable object should never be observed outside of the scope in which it was created.
- S 가 state 를 mutate 할 수 있는 능력을 의미하기 때문에 run 메서드는 protected 이다. (State 와 다르게)
- ST의 구현 디테일보다는 mutable state 의 scope 를 제한하기 위해 타입 시스템을 사용하는 아이디어가 중요하다. 
#### 14.2.2 An algebra of mutable reference
- ST 모나드를 정의했다. ST 모나드의 첫번째 응용은 mutable reference 에 관한 DSL 이다.
- 이 언어는 몇가지 primitives 를 가지는 combinator library 로 구성되고 
  변경 가능한 메모리 셀을 캡슐화하고 격리하는 이러한 참조에 대해 이야기하기 위해 이 언어는 아래와 같은 기본 명령어를 가져야 한다.
  - Allocate a new mutable cell
  - Write to a mutable cell
  - Read from a mutable cell
- DSL 로 STRef 를 정의 했다. ST<S, A> 는 local mutations 을 STRef<S,A> type 에 넣어서 observers 로부터 숨긴다.
- STRef 의 constructor 는 private 임으로 우리는 초기값을 인수로 전달하여 STRef 의 companion object 의 invoke 로만 STRef 객체를 얻을 수 있다.
  생성되는 ST action 의 타입은 ST<S, STRef<S, A>> 이고, S 를 사용해서 run 을 해야하만 STRef 를 얻을 수 있다. 이때 STRef 와 ST 의 S 는 같다.
#### 14.2.3 Running mutable state actions
- ST<S, STRef<S, Int>> 를 실행시켜서 STRef<S, Int> 를 얻을 수 있다면 STRef 안의 가변 변수를 외부에서 관찰 할 수 있고, 이것은 ST 의 invariant 를 위반한다.
  이를 타입 시스템을 통해 보장하기 위해서 RunnableST 로 ST 를 wrapping 하고 RunnableST 만 실행할 수 있도록 한다. 
  RunnableST 는 S 를 지움으로써 외부에서 STRef 를 사용할 수 없도록 한다. STRef 의 S 는 ST 의 S 와 일치해야 하는데 이것을 일치시킬 수 있는 방법이 없다.
#### 14.2.4 The mutable array represented as a data type for the ST monad
- array 더 나아가 map 을 지원하는 타입을 정의할 수 있다.
#### 14.2.5 A purely functional in-place quicksort
- quicksort 를 ST 를 사용해서 정의함으로 side effect 가 없다는 것을 보증할 수 있다.
## 14.3 Purity is contextual
### 14.3.1 Definition by example
- constructor 는 memory 상에 유니크한 오브젝트를 생성한 다음 그것의 레퍼런스를 돌려주는 effect 를 가지고 있는데 그렇다고 해서 constructor 를 
  사용하는 프로그램이 impure 하다고 할 수 있을까? 아니다 대부분의 프로그램은 레퍼런스를 관찰하지 않기 때문에 constructor 사용하는 프로그램이 side effect 가 없다고 말 할 수 있다.
  이처럼 referentially transparent 는 컨택스트와 관련이 있다.
- An expression e is referentially transparent with regard to a program p
  if every occurrence of e in p can be replaced by the result of evaluating e without affecting the meaning of p.
- kotlin 에서 evaluation 은 어떤 일반적인 형태로의 환산을 의미한다. 여기서 expression e 의 evaluation 을 val 에 할당 함으로 일반 적인 형태로 강제할 수 있다.
  `>>> val v = e`
### 14.3.2 What counts as a side effect?
- 무엇을 side effect 로 쳐야 할까? 프로그램의 정확도에 영향을 미친다면 그것을 side effect 로 보고 추적하는게 중요하다.
  정적 타입 정보로 이러한 것들을 추적할 수 있게 하기 위해서 우리는 A domain-specific language 를 정의 하고,
  컴파일러에 의해서 이러한 것들이 강제되도록 algebra 를 정의해야 한다.
