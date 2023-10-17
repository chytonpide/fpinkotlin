package chapter9.exercises.ex11

import arrow.core.None
import arrow.core.Option
import arrow.core.Some
import arrow.core.extensions.option.foldable.get
import arrow.core.toOption
import chapter7.sec1.r
import chapter9.solutions.ex11.State
import chapter9.solutions.final.Failure
import chapter9.solutions.final.Location
import chapter9.solutions.final.ParseError
import chapter9.solutions.final.Parser
import chapter9.solutions.final.ParserDsl
import chapter9.solutions.final.Success
import jdk.internal.org.jline.utils.Colors.s
import utils.SOLUTION_HERE

typealias State = Location
/*
data class ParseError(val stack: List<Pair<Location, String>>)
data class Success<out A>(val a:A, val consumed:Int): Result<A>()
data class Failure(val get: ParserError): Result<Nothing>()
data class Location(val input: String, val offset: Int = 0)

private fun Location.toError(msg: String) =
    ParserError(listOf(this to message))

 */
//tag::init1[]
abstract class Parser : ParserDsl<ParseError>() {

    override fun string(s: String): Parser<String> =
        { state: State ->
            when (val idx =
                firstNonMatchingIndex(state.input, s, state.offset)) {
                is None ->
                    Success(s, s.length)
                is Some ->
                    Failure(
                        state.advanceBy(idx.t).toError("'$s'"), // advanceBy(i) 는 i 만큼 offset 이 이동한 새로운 Location 을 생성하고,
                        // 이 새로운 Location 은 error 가 발생한 위치를 나타낸다.
                        idx.t != 0
                    )
            }
        }

    private fun firstNonMatchingIndex(
        s1: String,
        s2: String,
        offset: Int
    ): Option<Int> {
        var result = 0
        val chars1 = s1.substring(offset).toCharArray()
        val chars2 = s2.toCharArray()
        for(char1 in chars1) {
            for(char2 in chars2) {
                if(char1 != char2) {
                    break;
                }
                result += 1
            }
        }

        return if(result !=0) {
            Some(result + offset)
        } else {
            if (s1.length - offset >= s2.length) None
            else Some(s1.length - offset)
        }
    }

    private fun State.advanceBy(i: Int): State = State(this.input, this.offset + i)

    override fun regex(r: String): Parser<String> =
        { state: State ->
            val tmp = state.input.substring(state.offset)
            val match: Option<MatchResult> = tmp.findPrefixOf(r.toRegex())
            when (match) {
                is Some ->
                    Success(match.t.value, match.t.value.length)
                is None ->
                    Failure(state.advanceBy(0).toError("'$r'"))
            }
        }

    private fun String.findPrefixOf(r: Regex): Option<MatchResult> =
        r.find(this).toOption().filter { it.range.first == 0 }

    override fun <A> succeed(a: A): Parser<A> = { _ -> Success(a, 0) }

    override fun <A> slice(p: Parser<A>): Parser<String> = { state: State ->
        when(val result = p(state)) {
            is Success -> Success(state.slice(result.consumed), result.consumed)
            is Failure -> result
        }
    }

    private fun State.slice(n: Int): String = this.input.substring(this.offset, this.offset + n)

}
//end::init1[]
