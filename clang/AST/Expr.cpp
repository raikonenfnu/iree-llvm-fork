//===--- Expr.cpp - Expression AST Node Implementation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expr class and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/IdentifierTable.h"
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

StringLiteral::StringLiteral(const char *strData, unsigned byteLength, 
                             bool Wide, QualType t) : 
  Expr(StringLiteralClass, t) {
  // OPTIMIZE: could allocate this appended to the StringLiteral.
  char *AStrData = new char[byteLength];
  memcpy(AStrData, strData, byteLength);
  StrData = AStrData;
  ByteLength = byteLength;
  IsWide = Wide;
}

StringLiteral::~StringLiteral() {
  delete[] StrData;
}

bool UnaryOperator::isPostfix(Opcode Op) {
  switch (Op) {
  case PostInc:
  case PostDec:
    return true;
  default:
    return false;
  }
}

/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "sizeof" or "[pre]++".
const char *UnaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown unary operator");
  case PostInc: return "++";
  case PostDec: return "--";
  case PreInc:  return "++";
  case PreDec:  return "--";
  case AddrOf:  return "&";
  case Deref:   return "*";
  case Plus:    return "+";
  case Minus:   return "-";
  case Not:     return "~";
  case LNot:    return "!";
  case Real:    return "__real";
  case Imag:    return "__imag";
  case SizeOf:  return "sizeof";
  case AlignOf: return "alignof";
  case Extension: return "__extension__";
  }
}

//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//

CallExpr::CallExpr(Expr *fn, Expr **args, unsigned numargs, QualType t)
  : Expr(CallExprClass, t), Fn(fn), NumArgs(numargs) {
  Args = new Expr*[numargs];
  for (unsigned i = 0; i != numargs; ++i)
    Args[i] = args[i];
}

/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "<<=".
const char *BinaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown binary operator");
  case Mul:       return "*";
  case Div:       return "/";
  case Rem:       return "%";
  case Add:       return "+";
  case Sub:       return "-";
  case Shl:       return "<<";
  case Shr:       return ">>";
  case LT:        return "<";
  case GT:        return ">";
  case LE:        return "<=";
  case GE:        return ">=";
  case EQ:        return "==";
  case NE:        return "!=";
  case And:       return "&";
  case Xor:       return "^";
  case Or:        return "|";
  case LAnd:      return "&&";
  case LOr:       return "||";
  case Assign:    return "=";
  case MulAssign: return "*=";
  case DivAssign: return "/=";
  case RemAssign: return "%=";
  case AddAssign: return "+=";
  case SubAssign: return "-=";
  case ShlAssign: return "<<=";
  case ShrAssign: return ">>=";
  case AndAssign: return "&=";
  case XorAssign: return "^=";
  case OrAssign:  return "|=";
  case Comma:     return ",";
  }
}

/// isLvalue - C99 6.3.2.1: an lvalue is an expression with an object type or an
/// incomplete type other than void. Nonarray expressions that can be lvalues:
///  - name, where name must be a variable
///  - e[i]
///  - (e), where e must be an lvalue
///  - e.name, where e must be an lvalue
///  - e->name
///  - *e, the type of e cannot be a function type
///  - string-constant
///
bool Expr::isLvalue() {
  // first, check the type (C99 6.3.2.1)
  if (!TR->isObjectType())
    return false;
  if (TR->isIncompleteType() && TR->isVoidType())
    return false;
  
  // the type looks fine, now check the expression
  switch (getStmtClass()) {
  case StringLiteralClass: // C99 6.5.1p4
    return true;
  case ArraySubscriptExprClass: // C99 6.5.3p4 (e1[e2] == (*((e1)+(e2))))
    return true;
  case DeclRefExprClass: // C99 6.5.1p2
    return isa<VarDecl>(cast<DeclRefExpr>(this)->getDecl());
  case MemberExprClass: // C99 6.5.2.3p4
    const MemberExpr *m = cast<MemberExpr>(this);
    return m->isArrow() ? true : m->getBase()->isLvalue();
  case UnaryOperatorClass: // C99 6.5.3p4
    return cast<UnaryOperator>(this)->getOpcode() == UnaryOperator::Deref;
  case ParenExprClass: // C99 6.5.1p5
    return cast<ParenExpr>(this)->getSubExpr()->isLvalue();
  default: 
    return false;
  }
}

/// isModifiableLvalue - C99 6.3.2.1: an lvalue that does not have array type,
/// does not have an incomplete type, does not have a const-qualified type, and
/// if it is a structure or union, does not have any member (including, 
/// recursively, any member or element of all contained aggregates or unions)
/// with a const-qualified type.
bool Expr::isModifiableLvalue() {
  if (!isLvalue())
    return false;
  
  if (TR.isConstQualified())
    return false;
  if (TR->isArrayType())
    return false;
  if (TR->isIncompleteType())
    return false;
  if (const RecordType *r = dyn_cast<RecordType>(TR.getCanonicalType()))
    return r->isModifiableLvalue();
  return true;    
}

/// isConstantExpr - this recursive routine will test if an expression is
/// either a constant expression (isIntConst == false) or an integer constant
/// expression (isIntConst == true). Note: With the introduction of VLA's in
/// C99 the result of the sizeof operator is no longer always a constant
/// expression. The generalization of the wording to include any subexpression
/// that is not evaluated (C99 6.6p3) means that nonconstant subexpressions
/// can appear as operands to other operators (e.g. &&, ||, ?:). For instance,
/// "0 || f()" can be treated as a constant expression. In C90 this expression,
/// occurring in a context requiring a constant, would have been a constraint
/// violation. FIXME: This routine currently implements C90 semantics.
/// To properly implement C99 semantics this routine will need to evaluate
/// expressions involving operators previously mentioned.
bool Expr::isConstantExpr(bool isIntConst) const {
  switch (getStmtClass()) {
  case IntegerLiteralClass:
  case CharacterLiteralClass:
    return true;
  case FloatingLiteralClass:
  case StringLiteralClass:
    return isIntConst ? false : true;
  case DeclRefExprClass:
    return isa<EnumConstantDecl>(cast<DeclRefExpr>(this)->getDecl());
  case UnaryOperatorClass:
    const UnaryOperator *uop = cast<UnaryOperator>(this);
    if (uop->isIncrementDecrementOp()) // C99 6.6p3
      return false;
    // C99 6.5.3.4p2: otherwise, the operand is *not* evaluated and the result
    // is an integer constant. This effective ignores any subexpression that
    // isn't actually a constant expression (what an odd language:-)
    if (uop->isSizeOfAlignOfOp())
      return uop->getSubExpr()->getType()->isConstantSizeType();
    return uop->getSubExpr()->isConstantExpr(isIntConst);
  case BinaryOperatorClass:
    const BinaryOperator *bop = cast<BinaryOperator>(this);
    // C99 6.6p3: shall not contain assignment, increment/decrement,
    // function call, or comma operators, *except* when they are contained
    // within a subexpression that is not evaluated. 
    if (bop->isAssignmentOp() || bop->getOpcode() == BinaryOperator::Comma)
      return false;
    return bop->getLHS()->isConstantExpr(isIntConst) &&
           bop->getRHS()->isConstantExpr(isIntConst);
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->isConstantExpr();
  case CastExprClass: 
    const CastExpr *castExpr = cast<CastExpr>(this);    
    // C99 6.6p6: shall only convert arithmetic types to integer types.
    if (!castExpr->getSubExpr()->getType()->isArithmeticType())
      return false;
    if (!castExpr->getDestType()->isIntegerType())
      return false;      
    // allow floating constants that are the immediate operands of casts.
    if (castExpr->getSubExpr()->isConstantExpr() ||
        isa<FloatingLiteral>(castExpr->getSubExpr()))
      return true;
    return false;
  case SizeOfAlignOfTypeExprClass:
    const SizeOfAlignOfTypeExpr *sizeExpr = cast<SizeOfAlignOfTypeExpr>(this);
    if (sizeExpr->isSizeOf())
      return sizeExpr->getArgumentType()->isConstantSizeType();
    return true; // alignof will always evaluate to a constant
  case ConditionalOperatorClass:
    const ConditionalOperator *condExpr = cast<ConditionalOperator>(this);
    return condExpr->getCond()->isConstantExpr(isIntConst) &&
           condExpr->getLHS()->isConstantExpr(isIntConst) &&
           condExpr->getRHS()->isConstantExpr(isIntConst);
  default: 
    return false;
  }
}

bool Expr::isNullPointerConstant() const {
  const IntegerLiteral *constant = dyn_cast<IntegerLiteral>(this);
  if (!constant || constant->getValue() != 0)
    return false;
  return true;
}
