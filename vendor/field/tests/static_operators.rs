use field::*;

field::define_prime_field! {
    pub Small {limbs:1,modulus:[251],element:SmallElement,context:SmallField,}
}

#[test]
fn static_operators_and_context_calls_agree() {
    let field = SmallField::new();
    let a = field.from_integer(&24u64);
    let b = field.from_integer(&37u64);
    assert_eq!(a + b, field.add(&a, &b));
    assert_eq!(a - b, field.sub(&a, &b));
    assert_eq!(a * b, field.mul(&a, &b));
    assert_eq!(-a, field.neg(&a));
    let mut c = a;
    c += b;
    c *= b;
    c -= a;
    assert_eq!(c, (a + b) * b - a);
}

#[test]
fn static_field_power_inverse_and_division_cover_zero() {
    let field = SmallField::new();
    for a in 0u64..251 {
        let value = field.from_integer(&a);
        assert_eq!(field.pow_ct(&value, &Uint::<1>::ZERO), field.one());
        assert_eq!(
            field.pow_ct(&value, &Uint::from_words([3])),
            value * value * value
        );
        let inverse = field.inverse_ct(&value);
        assert_eq!(inverse.validity().declassify(), a != 0);
        if a != 0 {
            assert_eq!(value * *inverse.value(), field.one());
            assert_eq!(field.div_ct(&value, &value).value(), &field.one());
        }
    }
}
