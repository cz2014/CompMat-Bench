"""Mechanical workflow grading against explicitly accepted references."""
import argparse, json, math
from pathlib import Path

def finite(value):
    if type(value) not in (int,float):return False
    try:return math.isfinite(value)
    except OverflowError:return False

def numeric(value,spec):
    if not finite(value):return False
    truth=spec['truth'];rule=spec.get('rule','tolerance')
    if not finite(truth):raise ValueError('Unverified or invalid numerical reference')
    if rule=='integer':return int(value)==value and value==truth
    if rule=='sign':return value*truth>0 if truth else value==0
    if rule=='relative':return (value*truth>0 if truth else value==0) and abs(value-truth)<=spec['rtol']*abs(truth)
    return abs(value-truth)<=max(spec['atol'],spec['rtol']*abs(truth))

def pairs(values,allowed):
    if not isinstance(values,list) or not values:raise ValueError('Expected nonempty pair array')
    normalized=[]
    for label in values:
        if not isinstance(label,str):raise ValueError('Pair labels must be strings')
        atoms=label.split('-')
        if len(atoms)!=2 or not set(atoms)<=set(allowed):raise ValueError('Invalid magnetic site label')
        normalized.append('-'.join(sorted(atoms)))
    if len(set(normalized))!=len(normalized):raise ValueError('Duplicate undirected pair')
    return set(normalized)

def reference_approved(reference):
    return reference.get('status')=='verified' or (
        reference.get('chain')=='C09'
        and reference.get('status')=='approved_empirical'
        and reference.get('approval',{}).get('by')=='user'
        and reference.get('reference_scope')=='finite_duration_reproduction')

def grade(answer,reference,*,candidate=False):
    if not candidate and not reference_approved(reference):
        raise ValueError('Reference is not approved for production grading')
    if 'accepted_conventions' in reference:
        conventions=reference['accepted_conventions']
        if reference['chain']!='C01' or not isinstance(conventions,dict) or set(conventions)!={'original','rotated'}:
            raise ValueError('Invalid C01 convention references')
        if conventions['original']!=reference['fields']:
            raise ValueError('Original convention must retain the primary reference')
        for fields in conventions.values():
            if set(fields)!=set(reference['fields']):
                raise ValueError('Convention output schemas differ')
            for key,spec in fields.items():
                if {k:v for k,v in spec.items() if k!='truth'}!={k:v for k,v in reference['fields'][key].items() if k!='truth'}:
                    raise ValueError('Convention tolerances or rules differ')
        common={k:v for k,v in reference.items() if k!='accepted_conventions'}
        results={name:grade(answer,dict(common,fields=fields),candidate=candidate)
                 for name,fields in conventions.items()}
        matched=[name for name in ('original','rotated') if results[name]['score']==1]
        selected=matched[0] if matched else 'original'
        return dict(results[selected],matched_conventions=matched,
                    selected_convention=selected if matched else None,
                    convention_results=results)
    fields=reference['fields'];chain=reference['chain'];checks={}
    schema_ok=isinstance(answer,dict) and (set(fields)<=set(answer) if chain in ('C09','C10') else set(answer)==set(fields))
    if not schema_ok:
        return {'score':0,'checks':{'schema':False},'reference_status':reference['status']}
    if chain=='C02':
        rows=answer['families'];expected=fields['families'];seen={}
        if not isinstance(rows,list):return {'score':0,'checks':{'families_schema':False}}
        for row in rows:
            if not isinstance(row,dict) or set(row)!={'sg_number','count','Gamma_split_eV','generic_split_eV'}:
                return {'score':0,'checks':{'families_schema':False}}
            sg=row['sg_number']
            if not finite(sg) or int(sg)!=sg or not 1<=sg<=230 or sg in seen:
                return {'score':0,'checks':{'unique_space_groups':False}}
            seen[int(sg)]=row
        checks['family_set']=set(seen)==set(map(int,expected))
        for sg,specs in expected.items():
            row=seen.get(int(sg),{})
            for field,spec in specs.items():checks[f'{sg}.{field}']=numeric(row.get(field),spec)
    else:
        joint=reference.get('joint_sign_fields',[])
        for field,spec in fields.items():
            if field in joint:continue
            value=answer[field]
            if spec.get('rule')=='pairs':
                try:checks[field]=pairs(value,spec['allowed'])==pairs(spec['truth'],spec['allowed'])
                except ValueError:checks[field]=False
            elif spec.get('rule')=='boolean':checks[field]=type(value) is bool and value==spec['truth']
            else:checks[field]=numeric(value,spec)
        if joint:
            checks['joint_global_sign']=any(all(numeric(answer[k],dict(fields[k],truth=sign*fields[k]['truth'])) for k in joint) for sign in (1,-1))
    return {'score':int(all(checks.values())),'checks':checks,'reference_status':reference['status']}

def strict_load(path):
    def object_pairs(items):
        obj={}
        for key,value in items:
            if key in obj:raise ValueError('Duplicate JSON key')
            obj[key]=value
        return obj
    return json.loads(Path(path).read_text(),object_pairs_hook=object_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference');p.add_argument('answer');p.add_argument('--candidate',action='store_true');a=p.parse_args()
    ref=strict_load(a.reference)
    # Invalid submissions fail; invalid/unready references are configuration errors.
    if not reference_approved(ref) and not a.candidate:raise SystemExit('Reference not approved for production')
    try:result=grade(strict_load(a.answer),ref,candidate=a.candidate)
    except (OSError,ValueError) as exc:result={'score':0,'error':str(exc)}
    print(json.dumps(result,indent=2,allow_nan=False))
