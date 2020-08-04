package chapter12.solutions.ex11

import chapter11.Monad
import chapter12.Composite
import chapter12.CompositeOf
import chapter12.CompositePartialOf
import chapter12.fix

interface Listing<F, G> {

    //tag::init1[]
    fun <F, G> compose(
        mf: Monad<F>,
        mg: Monad<G>
    ): Monad<CompositePartialOf<F, G>> =
        object : Monad<CompositePartialOf<F, G>> {
            override fun <A> unit(a: A): CompositeOf<F, G, A> =
                Composite(mf.unit(mg.unit(a)))

            override fun <A, B> flatMap(
                mna: CompositeOf<F, G, A>,
                f: (A) -> CompositeOf<F, G, B>
            ): CompositeOf<F, G, B> =
                TODO("Simply can't be done!")

            override fun <A, B, C> compose(
                f: (A) -> CompositeOf<F, G, B>,
                g: (B) -> CompositeOf<F, G, C>
            ): (A) -> CompositeOf<F, G, C> = TODO()
        }
    //end::init1[]
}

/*
//tag::init2[]
fun <A, B> flatMap(
    mna: CompositeOf<F, G, A>,
    f: (A) -> CompositeOf<F, G, B>
): CompositeOf<F, G, B> =
        mf.flatMap(mna.fix().value) { na: Kind<G, A> ->
            mg.flatMap(na) { a: A ->
                f(a)
            }
        }
//end::init2[]
*/