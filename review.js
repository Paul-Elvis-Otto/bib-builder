/* Shared review controls for live results and persistent run history. */
function reviewNode(tag,text){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;return el;}
async function reviewRequest(url,options){const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data;}
function identificationText(counts){return `${counts.identified}/${counts.total} identified · ${counts.automatic} automatic · ${counts.confirmed} confirmed by you · ${counts.unresolved} unresolved · ${counts.errors} lookup errors`;}
function showIdentificationSummary(data){
    let panel=document.getElementById('identificationSummary');
    if(!panel){panel=reviewNode('div');panel.id='identificationSummary';document.getElementById('results').before(panel);}
    panel.replaceChildren();
    if(!data.counts)return;
    panel.append(reviewNode('h3',data.paper.title),reviewNode('p',identificationText(data.counts)));
    const meter=reviewNode('progress');meter.max=data.counts.total||1;meter.value=data.counts.identified;meter.setAttribute('aria-label','Identified references');panel.append(meter);
    const link=reviewNode('a','Open saved run');link.href='/history#'+encodeURIComponent(data.run_id);panel.append(link);
    panel.append(candidatePicker(data,-1,updated=>{showIdentificationSummary(updated);}));
}
function candidatePicker(data,index,onSaved){
    const match=index===-1?data.paper_match:data.matches[index];
    const original=index===-1?data.paper:data.original_entries[index];
    const box=reviewNode('details');box.open=index!==-1&&!match.work;
    box.append(reviewNode('summary',index===-1?'Review source paper identity':'Compare and select a match'));
    box.append(reviewNode('p',`Extracted: ${original.title} · ${(original.authors||[]).join('; ')} · ${original.year||'Year unknown'}${original.doi?' · DOI '+original.doi:''}`));
    const message=reviewNode('p',match.message||match.status);message.setAttribute('role','status');box.append(message);
    const candidates=[...(match.candidates||[])];
    if(match.work&&!candidates.some(c=>c.work.id===match.work.id))candidates.unshift({work:match.work});
    const buttons=[];
    async function save(workId){
        buttons.forEach(b=>b.disabled=true);message.textContent='Saving selection…';
        try{const updated=await reviewRequest('/api/choose-match',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id:data.run_id,index,work_id:workId})});onSaved(updated);}
        catch(error){message.textContent=error.message;buttons.forEach(b=>b.disabled=false);}
    }
    if(!candidates.length)box.append(reviewNode('p','No saved candidates. Re-extract this paper to search again.'));
    for(const c of candidates){
        const w=c.work;const card=reviewNode('div');card.className='candidate-card';
        card.append(reviewNode('h4',w.title||w.display_name||'Untitled'));
        card.append(reviewNode('p',[(w.authorships||[]).map(a=>a.author?.display_name).filter(Boolean).join('; '),w.publication_year,w.type,w.primary_location?.source?.display_name,w.doi].filter(Boolean).join(' · ')));
        if(c.title_similarity!==undefined)card.append(reviewNode('p',`Title similarity: ${Math.round(c.title_similarity*100)}% · Author: ${c.author_match?'agrees':'differs or missing'} · Year: ${c.year_match?'agrees':'differs or missing'}`));
        if(c.reasons?.length)card.append(reviewNode('p',c.reasons.join('; ')));
        if(/^https:\/\/openalex.org\/W\d+$/.test(w.id)){const link=reviewNode('a','View OpenAlex record');link.href=w.id;link.target='_blank';link.rel='noopener';card.append(link);}
        const chosen=match.work?.id===w.id;
        const button=reviewNode('button',chosen?(match.decision==='manual'?'Selected by you':'Confirm this automatic match'):'Use this match');button.type='button';button.disabled=chosen&&match.decision==='manual';button.onclick=()=>save(w.id);buttons.push(button);card.append(document.createTextNode(' '),button);box.append(card);
    }
    const clear=reviewNode('button','Leave unresolved / clear selection');clear.type='button';clear.onclick=()=>save(null);buttons.push(clear);box.append(clear);
    return box;
}
if(document.getElementById('historyView')){
    const list=document.getElementById('runList'),detail=document.getElementById('runDetail'),notice=document.getElementById('notice');
    let selected=null,generation=0;
    async function loadList(){const data=await reviewRequest('/api/runs');list.replaceChildren();if(!data.runs.length)list.append(reviewNode('p','No completed runs saved yet.'));
        let total=0,identified=0;data.runs.forEach(r=>{total+=r.counts.total;identified+=r.counts.identified;});
        list.append(reviewNode('p',`${data.runs.length} saved runs · ${identified}/${total} reference occurrences identified across runs`));
        data.runs.forEach(run=>{const card=reviewNode('article');const link=reviewNode('a',run.paper.title);link.href='#'+encodeURIComponent(run.run_id);card.append(link,reviewNode('p',run.created?new Date(run.created*1000).toLocaleString():'Imported earlier result — original run date unavailable'),reviewNode('p',identificationText(run.counts)));list.append(card);});
    }
    function renderRun(data){selected=data;detail.hidden=false;detail.replaceChildren();detail.append(reviewNode('h2',data.paper.title),reviewNode('p',identificationText(data.counts)));
        const progress=reviewNode('progress');progress.max=data.counts.total||1;progress.value=data.counts.identified;progress.setAttribute('aria-label','Identified references');detail.append(progress);
        addObsidianView(detail,data.entries,notice);
        const tools=reviewNode('div');tools.className='toolbar';
        for(const [format,label] of [['bibtex','Download BibTeX'],['ris','Download RIS']]){const link=reviewNode('a',label);link.href='/api/runs/'+encodeURIComponent(data.run_id)+'/'+format;tools.append(link);}
        const filter=reviewNode('select');filter.setAttribute('aria-label','Filter citations');for(const [value,label] of [['all','All citations'],['pending','Needs a decision'],['identified','Identified']]){const option=reviewNode('option',label);option.value=value;filter.append(option);}tools.append(filter);detail.append(tools);
        const onSaved=async updated=>{notice.textContent='Selection saved. Counts and exports updated.';renderRun(updated);try{await loadList();}catch(e){notice.textContent=e.message;}};
        detail.append(candidatePicker(data,-1,onSaved));
        const references=reviewNode('div');detail.append(references);
        function draw(){references.replaceChildren();data.entries.forEach((entry,index)=>{const matched=Boolean(data.matches[index].work);if(filter.value==='pending'&&matched||filter.value==='identified'&&!matched)return;
            const card=reviewNode('article');card.id='citation-'+index;card.append(reviewNode('h3',`${index+1}. ${entry.title}`),reviewNode('p',`${data.matches[index].decision==='manual'?'Confirmed by you':matched?'Automatic match':data.matches[index].status} · ${entry.authors.join('; ')} · ${entry.year||''}`));
            const original=reviewNode('details');original.append(reviewNode('summary','Original printed reference'),reviewNode('p',data.original_entries[index].raw_entry));card.append(original,candidatePicker(data,index,onSaved));
            const bib=reviewNode('textarea');bib.readOnly=true;bib.value=data.bibtex_records[index];bib.setAttribute('aria-label','BibTeX citation '+(index+1));card.append(bib);references.append(card);
        });}filter.onchange=draw;draw();
    }
    async function openHash(){const token=++generation;const [id,index]=location.hash.slice(1).split('/');if(!id){detail.hidden=true;return;}notice.textContent='Loading saved run…';try{const result=await reviewRequest('/api/runs/'+encodeURIComponent(decodeURIComponent(id)));if(token!==generation)return;renderRun(result);notice.textContent='';(document.getElementById('citation-'+index)||detail).scrollIntoView({block:'start'});}catch(e){if(token===generation)notice.textContent=e.message;}}
    document.getElementById('refreshRuns').onclick=()=>loadList().catch(e=>notice.textContent=e.message);
    window.addEventListener('hashchange',openHash);
    loadList().then(openHash).catch(e=>notice.textContent=e.message);
}
function obsidianLinks(entries){return entries.map(entry=>{const first=(entry.authors||[])[0]||'';const family=first.includes(',')?first.split(',',1)[0].trim():first.trim().split(/\s+/).filter(Boolean).at(-1)||'Unknown';const year=entry.year?String(entry.year):'nd';const title=String(entry.title||'Untitled').replace(/[\r\n]+/g,' ').trim();return `[[${family}${year}|${title}]]`;}).join('\n');}
async function copyObsidian(entries,textarea,status){const value=obsidianLinks(entries);textarea.value=value;textarea.hidden=false;try{await navigator.clipboard.writeText(value);status.textContent='Obsidian links copied to clipboard.';}catch(_){textarea.focus();textarea.select();status.textContent='Links are selected below. Press Ctrl+C or Command+C to copy.';}}
function addObsidianView(container,entries,status){const area=reviewNode('section');area.className='obsidian-export';const button=reviewNode('button','Copy Obsidian links');button.type='button';button.onclick=()=>copyObsidian(entries,textarea,status);const textarea=reviewNode('textarea');textarea.hidden=true;textarea.readOnly=true;textarea.setAttribute('aria-label','Obsidian links, one per line');area.append(button,textarea);container.append(area);}
